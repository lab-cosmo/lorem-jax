from dataclasses import dataclass

from jaxpme.batched_mixed.kspace import count_halfspace_kvectors
from marathon.data.properties import DEFAULT_PROPERTIES
from marathon.extra.hermes.pain import (
    MapTransform,
    Record,
)

NO_PADDING = ["k", "atoms_pbc", "pbc", "pairs_nonpbc"]


@dataclass(frozen=True)
class ToSample(MapTransform):
    cutoff: float
    keys: tuple | None = None
    properties: dict | None = None
    energy: bool = True
    forces: bool = True
    stress: bool = False
    lr_wavelength: float | None = None
    smearing: float | None = None

    def map(self, atoms):
        from lorem.batching import to_sample

        properties = self.properties if self.properties is not None else DEFAULT_PROPERTIES
        return to_sample(
            atoms,
            self.cutoff,
            keys=self.keys,
            energy=self.energy,
            forces=self.forces,
            stress=self.stress,
            lr_wavelength=self.lr_wavelength,
            smearing=self.smearing,
            properties=properties,
        )


@dataclass(frozen=True)
class ToBatch:
    batch_size: int
    keys: tuple = ("energy", "forces")
    properties: dict | None = None
    drop_remainder: bool = True
    size_strategy: str = "powers_of_4"
    fine_strategy: str = "powers_of_2"
    coarse_strategy: str = "powers_of_8"
    max_total_k: int = 2**42
    max_total_atoms: int = 2**42
    max_total_nobpc_pairs: int = 2**42

    def count_samples(self, batch):
        return batch.sr.structure_mask.sum()

    def info(self, batch):
        return _info(batch)

    def __call__(self, input_iterator):
        def k_size(rec):
            lr = rec.data.structure.get("lr", None)
            kg = getattr(lr, "k_grid", None)
            return 0 if kg is None else count_halfspace_kvectors(kg.shape)

        def nobpc_size(rec):
            lr = rec.data.structure.get("lr", None)
            return len(lr.centers) if hasattr(lr, "centers") else 0

        records, last_meta = [], None
        total_pbc_structures = 0
        max_k = 0
        total_atoms = 0
        total_nopbc_pairs = 0

        for rec in input_iterator:
            ks = k_size(rec)
            nopbc = nobpc_size(rec)
            atoms = rec.data.structure["positions"].shape[0]

            if nopbc > self.max_total_nobpc_pairs:
                raise ValueError(
                    f"sample exceeds max nopbc pair: {nopbc}>{self.max_total_nobpc_pairs}"
                )
            if ks > self.max_total_k:
                raise ValueError(f"sample exceeds max kgrid: {ks}>{self.max_total_k}")

            if atoms > self.max_total_atoms:
                raise ValueError(
                    f"sample exceeds max atoms: {atoms}>{self.max_total_atoms}"
                )

            # check limits BEFORE appending
            next_len = len(records) + 1
            next_pbc = total_pbc_structures + (1 if ks else 0)
            next_max_k = max(max_k, ks)
            next_total_nobpc_pairs = total_nopbc_pairs + nopbc
            next_atoms = total_atoms + atoms
            exceeds = (
                next_len >= self.batch_size
                or ((next_pbc * next_max_k) > self.max_total_k)
                or (next_atoms > self.max_total_atoms)
                or (next_total_nobpc_pairs > self.max_total_nobpc_pairs)
            )
            if records and exceeds:
                batch = self._batch(records)
                yield Record(last_meta.remove_record_key(), batch)
                records, last_meta = [], None
                total_pbc_structures = 0
                total_nopbc_pairs = 0
                max_k = 0
                total_atoms = 0

            records.append(rec.data)
            last_meta = rec.metadata
            if ks:
                total_pbc_structures += 1
                max_k = max(max_k, ks)

            total_nopbc_pairs += nopbc
            total_atoms += atoms

        if records and not self.drop_remainder:
            yield Record(last_meta.remove_record_key(), self._batch(records))

    def _batch(self, records_to_batch):
        from lorem.batching import to_batch

        properties = self.properties if self.properties is not None else DEFAULT_PROPERTIES
        return to_batch(
            records_to_batch,
            self.keys,
            batch_size=self.batch_size,
            strategies={
                "default": self.size_strategy,
                "fine": self.fine_strategy,
                "coarse": self.coarse_strategy,
            },
            properties=properties,
        )


def _info(batch):
    # non-padded entries
    real = {
        "pairs": batch.sr.pair_mask.sum(),
        "atoms": batch.sr.atom_mask.sum(),
        "samples": batch.sr.structure_mask.sum(),
        "pairs_nonpbc": batch.nopbc.pair_mask.sum(),
        "atoms_pbc": batch.pbc.atom_mask.sum(),
        "pbc": batch.pbc.structure_mask.sum(),
        "k": ((batch.pbc.k_grid != 0).any(axis=-1)).sum(),
    }

    # total entries w/ padding
    total = {
        "pairs": batch.sr.pair_mask.shape[0],
        "atoms": batch.sr.atom_mask.shape[0],
        "samples": batch.sr.structure_mask.shape[0],
        "pairs_nonpbc": batch.nopbc.centers.shape[0],
        "atoms_pbc": batch.pbc.atom_mask.size,
        "pbc": batch.pbc.structure_mask.shape[0],
        "k": batch.pbc.k_grid.shape[0] * batch.pbc.k_grid.shape[1],
    }

    # shapes relevant for JIT
    shape = {
        "samples": batch.sr.structure_mask.shape[0],
        "pairs": batch.sr.pair_mask.shape[0],
        "atoms": batch.sr.atom_mask.shape[0],
        "pairs_nonpbc": batch.nopbc.centers.shape[0],
        "atoms_pbc": batch.pbc.atom_mask.shape[1],
        "pbc": batch.pbc.structure_mask.shape[0],
        "k": batch.pbc.k_grid.shape[1],
    }

    return real, total, shape
