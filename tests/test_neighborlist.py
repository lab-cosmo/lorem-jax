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


def test_needs_update_cell_change(cache, ar_bulk):
    """Cell change always triggers update."""
    cache.save_reference(ar_bulk)

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
