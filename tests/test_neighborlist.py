import pytest
from ase.build import bulk

from lorem.neighborlist import NeighborListCache


@pytest.fixture
def ar_bulk():
    return bulk("Ar") * [2, 2, 2]


@pytest.fixture
def cache():
    return NeighborListCache(skin=0.5)


def test_needs_update_first_call(cache, ar_bulk):
    """First call always needs update (no reference stored)."""
    assert cache.needs_update(ar_bulk) is True


def test_needs_update_no_change(cache, ar_bulk):
    """No change after saving reference — no update needed."""
    cache.save_reference(ar_bulk)
    assert cache.needs_update(ar_bulk) is False


def test_needs_update_small_displacement(cache, ar_bulk):
    """Displacement below 0.5*skin — no update needed."""
    cache.save_reference(ar_bulk)

    displaced = ar_bulk.copy()
    # Move one atom by 0.1 Å (< 0.5 * 0.5 = 0.25 Å)
    pos = displaced.get_positions()
    pos[0, 0] += 0.1
    displaced.set_positions(pos)

    assert cache.needs_update(displaced) is False


def test_needs_update_large_displacement(cache, ar_bulk):
    """Displacement above 0.5*skin — update needed."""
    cache.save_reference(ar_bulk)

    displaced = ar_bulk.copy()
    # Move one atom by 0.3 Å (> 0.5 * 0.5 = 0.25 Å)
    pos = displaced.get_positions()
    pos[0, 0] += 0.3
    displaced.set_positions(pos)

    assert cache.needs_update(displaced) is True


def test_needs_update_boundary_displacement(cache, ar_bulk):
    """Displacement exactly at 0.5*skin boundary — no update needed (strict >)."""
    cache.save_reference(ar_bulk)

    displaced = ar_bulk.copy()
    pos = displaced.get_positions()
    # Exactly 0.25 Å displacement: (0.25)^2 = 0.0625, threshold = 0.0625
    # strict >, so this should NOT trigger update
    pos[0, 0] += 0.25
    displaced.set_positions(pos)

    assert cache.needs_update(displaced) is False


def test_needs_update_cell_change_no_shift_info(cache, ar_bulk):
    """Without max_cell_shift info, any cell change triggers update."""
    cache.save_reference(ar_bulk)  # no max_cell_shift → exact comparison

    modified = ar_bulk.copy()
    cell = modified.get_cell()
    cell[0, 0] += 0.01
    modified.set_cell(cell)

    assert cache.needs_update(modified) is True


def test_needs_update_pbc_change(cache, ar_bulk):
    """PBC change triggers update."""
    cache.save_reference(ar_bulk)

    modified = ar_bulk.copy()
    modified.set_pbc([True, True, False])

    assert cache.needs_update(modified) is True


def test_needs_update_natoms_change(cache, ar_bulk):
    """Number of atoms change triggers update."""
    cache.save_reference(ar_bulk)

    smaller = bulk("Ar") * [2, 2, 1]
    assert cache.needs_update(smaller) is True


def test_needs_update_numbers_change(cache, ar_bulk):
    """Atomic numbers change triggers update."""
    cache.save_reference(ar_bulk)

    modified = ar_bulk.copy()
    numbers = modified.get_atomic_numbers()
    numbers[0] = 36  # Change Ar (18) to Kr (36)
    modified.set_atomic_numbers(numbers)

    assert cache.needs_update(modified) is True


def test_save_reference_and_reset(cache, ar_bulk):
    """save_reference stores state, reset clears it."""
    cache.save_reference(ar_bulk)
    assert cache.needs_update(ar_bulk) is False

    cache.reset()
    assert cache.needs_update(ar_bulk) is True


def test_cumulative_displacement(ar_bulk):
    """Cumulative displacement from reference is what matters."""
    cache = NeighborListCache(skin=0.4)
    cache.save_reference(ar_bulk)

    displaced = ar_bulk.copy()
    pos = displaced.get_positions()

    # First displacement: 0.15 Å (< 0.5 * 0.4 = 0.2 Å) — ok
    pos[0, 0] += 0.15
    displaced.set_positions(pos)
    assert cache.needs_update(displaced) is False

    # Cumulative displacement: 0.25 Å (> 0.2 Å) — needs update
    pos[0, 0] += 0.10
    displaced.set_positions(pos)
    assert cache.needs_update(displaced) is True


def test_default_skin():
    """Default skin is 0.25 Å."""
    cache = NeighborListCache()
    assert cache.skin == 0.25


# -- Cell-change tests (combined position + cell criterion) --


def test_needs_update_small_cell_change_with_shift(ar_bulk):
    """Small cell change with max_cell_shift=1 stays within skin.

     cell₀ (reference)              cell = cell₀ * 1.001
    ┌──────────┐                   ┌───────────┐
    │  · · · · │   0.1% scaling    │  · · · ·  │
    │  · · · · │   ───────────>    │  · · · ·  │
    │  · · · · │   Δcell ≈ 0.005   │  · · · ·  │
    └──────────┘                   └───────────┘
    max_shift * Σ|Δcell_A| ≈ 0.016 Å  << skin = 0.5 Å
    """
    cache = NeighborListCache(skin=0.5)
    cache.save_reference(ar_bulk, max_cell_shift=1)

    scaled = ar_bulk.copy()
    scaled.set_cell(ar_bulk.get_cell() * 1.001, scale_atoms=True)

    assert cache.needs_update(scaled) is False


def test_needs_update_large_cell_change_with_shift(ar_bulk):
    """Large cell change with max_cell_shift=1 exceeds skin.

    10% cell scaling: Δcell_A ≈ 0.5 Å per vector
    max_shift * Σ|Δcell_A| ≈ 1 * 1.5 = 1.5 Å  >> skin = 0.5 Å
    """
    cache = NeighborListCache(skin=0.5)
    cache.save_reference(ar_bulk, max_cell_shift=1)

    scaled = ar_bulk.copy()
    scaled.set_cell(ar_bulk.get_cell() * 1.1, scale_atoms=True)

    assert cache.needs_update(scaled) is True


def test_needs_update_cell_change_zero_shift(ar_bulk):
    """With max_cell_shift=0, cell changes have no effect on criterion.

    Non-periodic systems have all cell_shifts = [0,0,0], so cell
    deformation doesn't change any pairwise distance.
    """
    cache = NeighborListCache(skin=0.5)
    cache.save_reference(ar_bulk, max_cell_shift=0)

    modified = ar_bulk.copy()
    cell = modified.get_cell()
    cell[0, 0] += 1.0  # large change, but shift=0 → no contribution
    modified.set_cell(cell)

    assert cache.needs_update(modified) is False


def test_needs_update_combined_position_and_cell(ar_bulk):
    """Position + cell change, each small, combined within skin.

    d_max = 0.05 Å  →  2*d_max = 0.1 Å
    Δcell ≈ 0.005 Å  →  max_shift * Σ|Δcell| ≈ 0.016 Å
    total ≈ 0.116 Å  <  skin = 0.5 Å  →  no rebuild
    """
    cache = NeighborListCache(skin=0.5)
    cache.save_reference(ar_bulk, max_cell_shift=1)

    modified = ar_bulk.copy()
    modified.set_cell(ar_bulk.get_cell() * 1.001, scale_atoms=True)
    pos = modified.get_positions()
    pos[0, 0] += 0.05
    modified.set_positions(pos)

    assert cache.needs_update(modified) is False


def test_needs_update_combined_exceeds_skin(ar_bulk):
    """Position + cell change that individually are small but combined exceed skin.

    skin = 0.2 Å
    d_max = 0.06 Å  →  2*d_max = 0.12 Å
    Δcell ≈ 0.15 Å  →  max_shift * Σ|Δcell| ≈ 0.09 Å
    total ≈ 0.21 Å  >  skin = 0.2 Å  →  rebuild!
    """
    cache = NeighborListCache(skin=0.2)
    cache.save_reference(ar_bulk, max_cell_shift=1)

    modified = ar_bulk.copy()
    # Cell change: scale by 1.01 → Δcell ≈ 0.05 per vector, 3 vectors
    modified.set_cell(ar_bulk.get_cell() * 1.01, scale_atoms=True)
    pos = modified.get_positions()
    pos[0, 0] += 0.06
    modified.set_positions(pos)

    assert cache.needs_update(modified) is True


def test_needs_update_higher_cell_shift(ar_bulk):
    """Higher max_cell_shift amplifies cell deformation contribution.

    For Ar FCC 2×2×2 (cell vectors ≈ 7.4 Å each), a 1% scaling gives:
    - Position d_max ≈ 0.09 Å (atom farthest from origin)
    - |Δcell_A| ≈ 0.074 Å per vector, Σ ≈ 0.22 Å

    Combined criterion: 2*d_max + max_shift * Σ|Δcell_A|
    - shift=1: 0.18 + 0.22 = 0.40  < 0.5 → ok
    - shift=3: 0.18 + 0.67 = 0.85  > 0.5 → rebuild
    """
    scaled = ar_bulk.copy()
    scaled.set_cell(ar_bulk.get_cell() * 1.01, scale_atoms=True)

    # With shift=1: within skin
    cache1 = NeighborListCache(skin=0.5)
    cache1.save_reference(ar_bulk, max_cell_shift=1)
    assert cache1.needs_update(scaled) is False

    # With shift=3: same deformation, amplified → exceeds skin
    cache3 = NeighborListCache(skin=0.5)
    cache3.save_reference(ar_bulk, max_cell_shift=3)
    assert cache3.needs_update(scaled) is True


def test_reset_clears_max_cell_shift(ar_bulk):
    """Reset clears max_cell_shift, reverting to exact cell comparison."""
    cache = NeighborListCache(skin=0.5)
    cache.save_reference(ar_bulk, max_cell_shift=1)

    scaled = ar_bulk.copy()
    scaled.set_cell(ar_bulk.get_cell() * 1.001, scale_atoms=True)

    # With shift info: small change tolerated
    assert cache.needs_update(scaled) is False

    # After reset + re-save without shift info: exact comparison
    cache.reset()
    cache.save_reference(ar_bulk)
    assert cache.needs_update(scaled) is True
