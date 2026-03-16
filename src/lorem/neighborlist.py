"""Verlet-style neighbor list cache.

Builds neighbor lists with cutoff + skin and reuses them as long as the
combined position displacement and cell deformation stays within the skin
budget. This avoids expensive neighbor searches on every MD step while
guaranteeing that all pairs within the physical cutoff are present in the
cached list.

For a pair (i, j) with periodic image shift S, the change in pairwise
distance from the reference is bounded by:

    |dR_ij| <= |dR_i| + |dR_j| + |S . d_cell|
            <= 2 * d_max + max_shift * sum(|d_cell_A|)

The neighbor list remains valid as long as this is < skin.
"""

import numpy as np


class NeighborListCache:
    """Cache for neighbor lists with skin-based recomputation.

    The Verlet criterion ensures that a neighbor list built with
    cutoff + skin contains all pairs within cutoff even after atomic
    displacements and cell deformations, as long as the combined
    change stays within the skin budget.

    Parameters
    ----------
    skin : float
        Skin distance in Angstrom. Default 0.25.
    """

    def __init__(self, skin=0.25):
        self.skin = skin
        self._reference_positions = None
        self._reference_cell = None
        self._reference_pbc = None
        self._reference_numbers = None
        self._max_cell_shift = None

    def needs_update(self, atoms):
        """Check if neighbor list needs recomputation.

        Returns True on first call, on any structural change (pbc,
        natoms, atomic numbers), or when the combined position
        displacement + cell deformation exceeds the skin budget.

        When max_cell_shift is not set (direct use without calculator),
        falls back to exact cell comparison for backward compatibility.
        """
        if self._reference_positions is None:
            return True

        if len(atoms) != len(self._reference_positions):
            return True
        if (atoms.get_atomic_numbers() != self._reference_numbers).any():
            return True
        if (atoms.get_pbc() != self._reference_pbc).any():
            return True

        # Cell handling depends on whether we have cell shift info
        if self._max_cell_shift is None:
            # No shift info — exact cell comparison (conservative)
            if (atoms.get_cell()[:] != self._reference_cell).any():
                return True

        # Position displacement: max over atoms of |dR|
        displacements = atoms.get_positions() - self._reference_positions
        max_disp = np.sqrt((displacements**2).sum(axis=1).max())

        # Cell deformation contribution
        if self._max_cell_shift is not None and self._max_cell_shift > 0:
            #   |S . d_cell| <= max_shift * sum_A(|d_cell_A|)
            # where |d_cell_A| is the norm of the change in cell vector A
            cell_change = atoms.get_cell()[:] - self._reference_cell
            cell_vector_norms = np.linalg.norm(cell_change, axis=1)
            max_cell_contrib = self._max_cell_shift * cell_vector_norms.sum()
        else:
            max_cell_contrib = 0.0

        # Combined criterion:
        #   max |dR_ij| <= 2*d_max + max_cell_contrib < skin
        return bool(2 * max_disp + max_cell_contrib > self.skin)

    def save_reference(self, atoms, max_cell_shift=None):
        """Store reference state after neighbor list rebuild.

        Parameters
        ----------
        atoms : ase.Atoms
            Reference atomic configuration.
        max_cell_shift : int or None
            Maximum absolute value of any cell shift component in the
            neighbor list. Enables the combined position + cell Verlet
            criterion. When None, falls back to exact cell comparison.
        """
        self._reference_positions = atoms.get_positions().copy()
        self._reference_cell = np.array(atoms.get_cell()[:]).copy()
        self._reference_pbc = atoms.get_pbc().copy()
        self._reference_numbers = atoms.get_atomic_numbers().copy()
        self._max_cell_shift = max_cell_shift

    def reset(self):
        """Clear the cache."""
        self._reference_positions = None
        self._reference_cell = None
        self._reference_pbc = None
        self._reference_numbers = None
        self._max_cell_shift = None
