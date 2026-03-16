"""Verlet-style neighbor list cache.

Builds neighbor lists with cutoff + skin and reuses them as long as no atom
has moved more than 0.5 * skin from its reference position. This avoids
expensive neighbor searches on every MD step while guaranteeing that all
pairs within the physical cutoff are present in the cached list.
"""

import numpy as np


class NeighborListCache:
    """Cache for neighbor lists with skin-based recomputation.

    The Verlet criterion: after both atoms of a pair move by at most
    skin/2, the pairwise distance changes by at most skin. So a neighbor
    list built with cutoff + skin contains all pairs within cutoff even
    after displacements up to skin/2.

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

    def needs_update(self, atoms):
        """Check if neighbor list needs recomputation.

        Returns True on first call, on any structural change (cell, pbc,
        natoms, atomic numbers), or when the maximum atomic displacement
        from the reference positions exceeds 0.5 * skin.
        """
        if self._reference_positions is None:
            return True

        if len(atoms) != len(self._reference_positions):
            return True
        if (atoms.get_atomic_numbers() != self._reference_numbers).any():
            return True
        if (atoms.get_cell()[:] != self._reference_cell).any():
            return True
        if (atoms.get_pbc() != self._reference_pbc).any():
            return True

        displacements = atoms.get_positions() - self._reference_positions
        max_sq_disp = (displacements**2).sum(axis=1).max()
        return bool(max_sq_disp > (0.5 * self.skin) ** 2)

    def save_reference(self, atoms):
        """Store reference state after neighbor list rebuild."""
        self._reference_positions = atoms.get_positions().copy()
        self._reference_cell = np.array(atoms.get_cell()[:]).copy()
        self._reference_pbc = atoms.get_pbc().copy()
        self._reference_numbers = atoms.get_atomic_numbers().copy()

    def reset(self):
        """Clear the cache."""
        self._reference_positions = None
        self._reference_cell = None
        self._reference_pbc = None
        self._reference_numbers = None
