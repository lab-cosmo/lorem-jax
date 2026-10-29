"""Long-range kernels: jax-pme Ewald backends behind one interface.

A batch keeps the geometry shared by the short-range and long-range parts
(``batch.geometry``) apart from the inputs private to the long-range kernel
(``batch.lr``). A kernel has no parameters, only configuration, so a trained
model runs with any of them: the choice changes convergence and speed, not
the model.

Host side, a kernel prepares one structure (``prepare``), reports what it adds
to a batch (``sample_shapes``) and collates prepared structures into the atom
and structure slots that ``lorem.batching.to_batch`` fixes (``batch``). On
device, ``potentials`` rebuilds the jax-pme batch from the shared geometry and
the kernel's inputs, picking the backend from the type of the inputs.
"""

import numpy as np

from collections import namedtuple
from dataclasses import dataclass

from jaxpme import batched_mixed, batched_tiled
from jaxpme.batched_mixed.batching import Batch as JaxpmeBatch

# real-space Ewald pairs (indexing the shared atom/structure slots), per-structure
# smearing and k-space cell, and the backend's non-periodic and periodic inputs
_FIELDS = (
    "centers",
    "others",
    "cell_shifts",
    "pair_mask",
    "pair_to_structure",
    "smearing",
    "effective_cell",
    "nopbc",
    "pbc",
)
EwaldInputs = namedtuple("EwaldInputs", _FIELDS)
TiledEwaldInputs = namedtuple("TiledEwaldInputs", _FIELDS)

_CALCULATORS = {
    EwaldInputs: batched_mixed.Ewald,
    TiledEwaldInputs: batched_tiled.Ewald,
}


@dataclass(frozen=True)
class Ewald:
    """Batched Ewald summation (``jaxpme.batched_mixed``).

    The default kernel. With ``num_k``, the k-grid has a fixed size and the
    real-space cutoff and smearing follow from it, independent of the model
    cutoff. Otherwise they follow from ``cutoff``, or from the model cutoff if
    that is not set either.
    """

    num_k: int | None = None
    cutoff: float | None = None

    def prepare(self, atoms, cutoff, skin=0.0):
        if self.num_k is None and self.cutoff is None:
            return _prepare(atoms, cutoff, None, skin)
        return _prepare(atoms, self.cutoff, self.num_k, skin)

    def sample_shapes(self, structure):
        return _sample_shapes(structure)

    def batch(self, structures, num_structures, num_atoms, strategies, shapes=None):
        from jaxpme.batched_mixed.batching import get_batch

        return _collate(
            EwaldInputs,
            get_batch(
                structures,
                num_structures=num_structures,
                num_atoms=num_atoms,
                **_sizes(strategies, shapes),
            ),
        )

    def info(self, inputs):
        pbc = inputs.pbc
        real = {
            "atoms_pbc": pbc.atom_mask.sum(),
            "pbc": pbc.structure_mask.sum(),
            "k": ((pbc.k_grid != 0).any(axis=-1)).sum(),
        }
        total = {
            "atoms_pbc": pbc.atom_mask.size,
            "pbc": pbc.structure_mask.shape[0],
            "k": pbc.k_grid.shape[0] * pbc.k_grid.shape[1],
        }
        shape = {
            "atoms_pbc": pbc.atom_mask.shape[1],
            "pbc": pbc.structure_mask.shape[0],
            "k": pbc.k_grid.shape[1],
        }
        return _merge_info(_pairs_info(inputs), (real, total, shape))


@dataclass(frozen=True)
class TiledEwald:
    """Batched Ewald with tile-dispatched k-space (``jaxpme.batched_tiled``).

    Built for training on heterogeneous batches: atoms of periodic structures
    are packed into ``BM``-sized tiles and k-vectors into ``BK``-sized tiles.
    ``num_k`` is required and sets the real-space cutoff and smearing;
    ``cutoff`` overrides only the real-space cutoff.
    """

    num_k: int
    cutoff: float | None = None
    BM: int = 32
    BK: int = 128

    def prepare(self, atoms, cutoff, skin=0.0):
        # the model cutoff plays no part: the k-grid sets the Ewald parameters
        return _prepare(atoms, self.cutoff, self.num_k, skin)

    def sample_shapes(self, structure):
        return _sample_shapes(structure)

    def batch(self, structures, num_structures, num_atoms, strategies, shapes=None):
        from jaxpme.batched_tiled.batching import get_batch

        return _collate(
            TiledEwaldInputs,
            get_batch(
                structures,
                num_structures=num_structures,
                num_atoms=num_atoms,
                BM=self.BM,
                BK=self.BK,
                **_sizes(strategies, shapes),
            ),
        )

    def info(self, inputs):
        pbc = inputs.pbc
        real = {
            "atoms_pbc": pbc.pbc_atom_mask.sum(),
            "pbc": pbc.structure_mask.sum(),
            "k": ((pbc.k_grid != 0).any(axis=-1)).sum(),
        }
        total = {
            "atoms_pbc": pbc.pbc_atom_mask.shape[0],
            "pbc": pbc.structure_mask.shape[0],
            "k": pbc.k_grid.shape[0] * pbc.k_grid.shape[1],
        }
        shape = {
            "atoms_pbc": pbc.pbc_atom_mask.shape[0],
            "pbc": pbc.structure_mask.shape[0],
            "k": pbc.k_grid.shape[1],
        }
        return _merge_info(_pairs_info(inputs), (real, total, shape))


def get_kernel(lr_kernel=None):
    """``lr_kernel``, or the default: Ewald with the model cutoff."""
    return Ewald() if lr_kernel is None else lr_kernel


def potentials(charges, geometry, inputs):
    """Long-range potentials of ``charges``, by the backend that made ``inputs``."""
    calculator = _CALCULATORS[type(inputs)]()
    return calculator.potentials(
        charges, to_jaxpme(geometry, inputs), inputs.nopbc, inputs.pbc
    )


def to_jaxpme(geometry, inputs):
    """Rebuild jax-pme's batch: shared geometry plus the kernel's own pairs."""
    return JaxpmeBatch(
        positions=geometry.positions,
        cell=geometry.cell,
        pbc=geometry.pbc,
        atom_mask=geometry.atom_mask,
        structure_mask=geometry.structure_mask,
        atom_to_structure=geometry.atom_to_structure,
        # 1D pbc is rejected in `prepare`, so periodic means 2D or 3D
        pbc_mask=geometry.pbc.any(axis=-1),
        centers=inputs.centers,
        others=inputs.others,
        cell_shifts=inputs.cell_shifts,
        pair_mask=inputs.pair_mask,
        pair_to_structure=inputs.pair_to_structure,
        smearing=inputs.smearing,
        effective_cell=inputs.effective_cell,
        distances=None,
    )


def max_cell_shift(inputs):
    return int(np.abs(np.array(inputs.cell_shifts)).max())


def update_effective_cell(inputs, cell, pbc, positions):
    """Recompute the k-space cell of a single-structure batch after a move.

    For 2D pbc, the vacuum vector is shrunk around the slab, so it follows the
    positions; for 3D pbc, the k-space cell is the cell itself. Non-periodic
    structures have no k-space part.
    """
    from jaxpme.batched_mixed.batching import shrink_2d_cell

    effective_cell = np.array(inputs.effective_cell)
    if pbc.sum() == 2:
        effective_cell[0] = shrink_2d_cell(cell, pbc, positions)
    elif pbc.all():
        effective_cell[0] = cell
    return inputs._replace(effective_cell=effective_cell)


# -- helpers --


def _prepare(atoms, cutoff, num_k, skin, dtype=np.float32):
    """Per-structure Ewald inputs, following jax-pme's conventions.

    ``num_k`` sets ``lr_wavelength`` (from the k-space cell); otherwise it is
    ``cutoff / 8``. Smearing is twice ``lr_wavelength`` and the default
    real-space cutoff eight times. ``skin`` widens only the neighbor list, so
    the Ewald split itself does not change. Non-periodic structures get the
    bare 1/r sum over all pairs and no real-space neighbor list.
    """
    from jaxpme.batched_mixed.batching import shrink_2d_cell, to_lr, to_structure
    from jaxpme.kspace import lr_wavelength_for_num_k

    pbc = atoms.get_pbc()

    if not pbc.any():
        structure = to_structure(atoms, None, dtype=dtype)
        _, structure["lr"] = to_lr(structure, None, None)
        return structure

    cell = atoms.get_cell().array.astype(dtype)
    if pbc.sum() == 2:
        effective_cell = shrink_2d_cell(cell, pbc, atoms.get_positions().astype(dtype))
    else:
        effective_cell = cell

    if num_k is not None:
        lr_wavelength = lr_wavelength_for_num_k(effective_cell, num_k)
        if cutoff is None:
            cutoff = lr_wavelength * 8.0
    elif cutoff is not None:
        lr_wavelength = cutoff / 8.0
    else:
        raise ValueError("one of cutoff or num_k is required")

    structure = to_structure(atoms, cutoff + skin, dtype=dtype)
    structure["effective_cell"] = effective_cell
    structure["smearing"], structure["lr"] = to_lr(
        structure, lr_wavelength, lr_wavelength * 2.0
    )

    return structure


def _sample_shapes(structure):
    lr = structure["lr"]
    is_pbc = hasattr(lr, "k_grid")
    return {
        "lr_pairs": len(structure["centers"]),
        "k": lr.k_grid.shape[0] if is_pbc else 0,
        "pairs_nonpbc": 0 if is_pbc else len(lr.centers),
    }


def _sizes(strategies, shapes):
    if shapes is None:
        default = strategies.get("default", "powers_of_2")
        coarse = strategies.get("coarse", default)
        return {
            "strategy": default,
            "num_structures_pbc": strategies.get("fine", default),
            "num_pairs": coarse,
            "num_pairs_nonpbc": coarse,
        }

    return {
        "strategy": "multiples",
        "num_structures_pbc": shapes["pbc"],
        "num_atoms_pbc": shapes["atoms_pbc"],
        # before the split, SR and LR shared one neighbor list
        "num_pairs": shapes.get("lr_pairs", shapes["pairs"]),
        "num_pairs_nonpbc": shapes["pairs_nonpbc"],
        "num_k": shapes["k"],
    }


def _collate(cls, jaxpme_batch):
    _, batch, nopbc, pbc = jaxpme_batch
    return cls(
        centers=batch.centers,
        others=batch.others,
        cell_shifts=batch.cell_shifts,
        pair_mask=batch.pair_mask,
        pair_to_structure=batch.pair_to_structure,
        smearing=batch.smearing,
        effective_cell=batch.effective_cell,
        nopbc=nopbc,
        pbc=pbc,
    )


def _pairs_info(inputs):
    real = {
        "lr_pairs": inputs.pair_mask.sum(),
        "pairs_nonpbc": inputs.nopbc.pair_mask.sum(),
    }
    total = {
        "lr_pairs": inputs.pair_mask.shape[0],
        "pairs_nonpbc": inputs.nopbc.centers.shape[0],
    }
    return real, total, dict(total)


def _merge_info(a, b):
    return tuple({**x, **y} for x, y in zip(a, b))
