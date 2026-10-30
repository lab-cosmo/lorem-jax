from dataclasses import dataclass
from functools import partial

from marathon.data.properties import DEFAULT_PROPERTIES
from marathon.grain import Record
from marathon.grain import ToSample as MarathonToSample

from lorem.batching import (
    check_shapes,
    needed_shapes,
    overflow,
    padded_shapes,
    to_structure,
)
from lorem.batching import sample_shapes as _sample_shapes
from lorem.longrange import get_kernel

NO_PADDING = ["k", "atoms_pbc", "pbc", "pairs_nonpbc"]


def ToSample(lr_kernel=None, **kwargs):
    # marathon's ToSample with lorem's geometry builder; labels stay float64 for stats
    kwargs.setdefault("float_dtype", "float64")
    structure_fn = partial(to_structure, lr_kernel=lr_kernel)
    return MarathonToSample(structure_fn=structure_fn, **kwargs)


@dataclass(frozen=True)
class ToBatch:
    """Group samples into batches, sized per axis (see lorem.batching).

    By default, each batch is sized to what it holds: every axis is rounded up
    by its size strategy, and the max_total_* caps bound what a batch may
    allocate (axes are clamped to their caps, and k-vectors and periodic atoms
    are capped as padded arrays). With ``shapes``, every batch has exactly
    those shapes, so a run compiles once; samples join a batch while they fit,
    and a sample that fits no batch raises.
    """

    batch_size: int
    keys: tuple = ("energy", "forces")
    inputs: tuple = ()
    properties: dict | None = None
    drop_remainder: bool = True
    size_strategy: str = "powers_of_4"
    fine_strategy: str = "powers_of_2"
    coarse_strategy: str = "powers_of_8"
    k_strategy: str | None = None  # None: the kernel's default, else size_strategy
    lr_pairs_strategy: str | None = None  # None: the kernel's, else coarse_strategy
    max_total_k: int = 2**42
    max_total_atoms: int = 2**42
    max_total_pairs: int = 2**42
    max_total_lr_pairs: int = 2**42
    max_total_nobpc_pairs: int = 2**42
    max_total_atoms_pbc: int = 2**42
    shapes: dict | None = None  # fixed shapes for every batch, one per axis
    lr_kernel: object = None  # None: lorem.longrange.Ewald()

    def __post_init__(self):
        object.__setattr__(self, "lr_kernel", get_kernel(self.lr_kernel))
        if self.shapes is not None:
            object.__setattr__(self, "shapes", check_shapes(self.shapes, self.lr_kernel))

    @property
    def strategies(self):
        strategies = {
            "default": self.size_strategy,
            "fine": self.fine_strategy,
            "coarse": self.coarse_strategy,
        }
        if self.k_strategy is not None:
            strategies["k"] = self.k_strategy
        if self.lr_pairs_strategy is not None:
            strategies["lr_pairs"] = self.lr_pairs_strategy
        return strategies

    @property
    def caps(self):
        return {
            "atoms": self.max_total_atoms,
            "pairs": self.max_total_pairs,
            "lr_pairs": self.max_total_lr_pairs,
            "pairs_nonpbc": self.max_total_nobpc_pairs,
        }

    def count_samples(self, batch):
        return batch.geometry.structure_mask.sum()

    def info(self, batch):
        return _info(batch, self.lr_kernel)

    def batch_shapes(self, sample_shapes):
        """Shapes of a batch of samples with these shapes (see lorem.batching)."""
        if self.shapes is not None:
            return self.shapes
        needed = needed_shapes(sample_shapes, self.lr_kernel)
        return padded_shapes(needed, self.strategies, self.lr_kernel, caps=self.caps)

    def overflow(self, sample_shapes):
        """Where a batch of these samples would exceed the budget, if anywhere."""
        too_large = {}
        if len(sample_shapes) + 1 > self.batch_size:  # +1: the padding structure
            too_large["structures"] = (len(sample_shapes) + 1, self.batch_size)

        needed = needed_shapes(sample_shapes, self.lr_kernel)
        if self.shapes is not None:
            return {**too_large, **overflow(needed, self.shapes)}

        too_large.update(overflow(needed, self.caps))
        if not too_large:
            # these are padded per periodic structure, so cap what is allocated
            allocated = self.lr_kernel.array_sizes(self.batch_shapes(sample_shapes))
            caps = {"k": self.max_total_k, "atoms_pbc": self.max_total_atoms_pbc}
            too_large.update(overflow(allocated, caps))

        return too_large

    def __call__(self, input_iterator):
        records, sample_shapes, last_meta = [], [], None

        for rec in input_iterator:
            shapes = _sample_shapes(rec.data.structure, self.lr_kernel)

            if too_large := self.overflow([shapes]):
                raise ValueError(f"sample exceeds the batch budget: {too_large}")

            if records and self.overflow(sample_shapes + [shapes]):
                batch = self._batch(records, sample_shapes)
                yield Record(last_meta.remove_record_key(), batch)
                records, sample_shapes, last_meta = [], [], None

            records.append(rec.data)
            sample_shapes.append(shapes)
            last_meta = rec.metadata

        if records and not self.drop_remainder:
            batch = self._batch(records, sample_shapes)
            yield Record(last_meta.remove_record_key(), batch)

    def _batch(self, records_to_batch, sample_shapes):
        from lorem.batching import to_batch

        properties = self.properties if self.properties is not None else DEFAULT_PROPERTIES
        return to_batch(
            records_to_batch,
            self.keys,
            inputs=self.inputs,
            batch_size=self.batch_size,
            shapes=self.batch_shapes(sample_shapes),
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
