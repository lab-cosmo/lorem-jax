from dataclasses import dataclass
from functools import partial

from marathon.data.properties import DEFAULT_PROPERTIES
from marathon.grain import Record
from marathon.grain import ToSample as MarathonToSample

from lorem.batching import to_structure
from lorem.longrange import get_kernel

NO_PADDING = ["k", "atoms_pbc", "pbc", "pairs_nonpbc"]


def ToSample(lr_kernel=None, **kwargs):
    # marathon's ToSample with lorem's geometry builder; labels stay float64 for stats
    kwargs.setdefault("float_dtype", "float64")
    structure_fn = partial(to_structure, lr_kernel=lr_kernel)
    return MarathonToSample(structure_fn=structure_fn, **kwargs)


@dataclass(frozen=True)
class ToBatch:
    batch_size: int
    keys: tuple = ("energy", "forces")
    inputs: tuple = ()
    properties: dict | None = None
    drop_remainder: bool = True
    size_strategy: str = "powers_of_4"
    fine_strategy: str = "powers_of_2"
    coarse_strategy: str = "powers_of_8"
    max_total_k: int = 2**42
    max_total_atoms: int = 2**42
    max_total_pairs: int = 2**42
    max_total_lr_pairs: int = 2**42
    max_total_nobpc_pairs: int = 2**42
    lr_kernel: object = None  # None: lorem.longrange.Ewald()

    def __post_init__(self):
        object.__setattr__(self, "lr_kernel", get_kernel(self.lr_kernel))

    def count_samples(self, batch):
        return batch.geometry.structure_mask.sum()

    def info(self, batch):
        return _info(batch, self.lr_kernel)

    def __call__(self, input_iterator):
        # budgets for sizes that add up over samples; k is budgeted on its own,
        # since every periodic structure is padded to the largest k-grid
        budgets = {
            "atoms": self.max_total_atoms,
            "pairs": self.max_total_pairs,
            "lr_pairs": self.max_total_lr_pairs,
            "pairs_nonpbc": self.max_total_nobpc_pairs,
        }

        records, last_meta = [], None
        totals = dict.fromkeys(budgets, 0)
        total_pbc_structures = 0
        max_k = 0

        for rec in input_iterator:
            structure = rec.data.structure
            sizes = {
                "atoms": structure["positions"].shape[0],
                "pairs": structure["centers"].shape[0],
                **self.lr_kernel.sample_shapes(structure["lr"]),
            }
            ks = sizes["k"]

            for key, budget in budgets.items():
                if sizes[key] > budget:
                    raise ValueError(f"sample exceeds max {key}: {sizes[key]}>{budget}")
            if ks > self.max_total_k:
                raise ValueError(f"sample exceeds max kgrid: {ks}>{self.max_total_k}")

            # check limits BEFORE appending
            next_len = len(records) + 1
            next_pbc = total_pbc_structures + (1 if ks else 0)
            next_max_k = max(max_k, ks)
            exceeds = (
                next_len >= self.batch_size
                or ((next_pbc * next_max_k) > self.max_total_k)
                or any(totals[k] + sizes[k] > budget for k, budget in budgets.items())
            )
            if records and exceeds:
                batch = self._batch(records)
                yield Record(last_meta.remove_record_key(), batch)
                records, last_meta = [], None
                totals = dict.fromkeys(budgets, 0)
                total_pbc_structures = 0
                max_k = 0

            records.append(rec.data)
            last_meta = rec.metadata
            if ks:
                total_pbc_structures += 1
                max_k = max(max_k, ks)

            for key in budgets:
                totals[key] += sizes[key]

        if records and not self.drop_remainder:
            yield Record(last_meta.remove_record_key(), self._batch(records))

    def _batch(self, records_to_batch):
        from lorem.batching import to_batch

        properties = self.properties if self.properties is not None else DEFAULT_PROPERTIES
        return to_batch(
            records_to_batch,
            self.keys,
            inputs=self.inputs,
            batch_size=self.batch_size,
            strategies={
                "default": self.size_strategy,
                "fine": self.fine_strategy,
                "coarse": self.coarse_strategy,
            },
            properties=properties,
            lr_kernel=self.lr_kernel,
        )


def _info(batch, lr_kernel):
    geometry, sr = batch.geometry, batch.sr

    # non-padded entries
    real = {
        "pairs": sr.pair_mask.sum(),
        "atoms": geometry.atom_mask.sum(),
        "samples": geometry.structure_mask.sum(),
    }

    # total entries w/ padding
    total = {
        "pairs": sr.pair_mask.shape[0],
        "atoms": geometry.atom_mask.shape[0],
        "samples": geometry.structure_mask.shape[0],
    }

    # shapes relevant for JIT
    shape = {
        "samples": geometry.structure_mask.shape[0],
        "pairs": sr.pair_mask.shape[0],
        "atoms": geometry.atom_mask.shape[0],
    }

    lr_real, lr_total, lr_shape = lr_kernel.info(batch.lr)

    return {**real, **lr_real}, {**total, **lr_total}, {**shape, **lr_shape}
