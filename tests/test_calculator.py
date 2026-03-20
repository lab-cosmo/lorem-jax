import numpy as np

from ase.build import bulk

from lorem.calculator import Calculator
from lorem.models.bec import LoremBEC
from lorem.models.mlip import Lorem


def test_calculator_energy_forces():
    model = Lorem(cutoff=5.0, num_features=8, num_spherical_features=2, num_radial=4)
    calc = Calculator.from_model(model)

    atoms = bulk("Ar") * [2, 2, 2]
    calc.calculate(atoms)

    assert "energy" in calc.results
    assert "forces" in calc.results
    assert calc.results["forces"].shape == (len(atoms), 3)
    assert "BEC" not in calc.results  # Lorem (not LoremBEC) has no BEC


def test_calculator_bec():
    model = LoremBEC(cutoff=5.0, num_features=8, num_spherical_features=2, num_radial=4)
    calc = Calculator.from_model(model)

    atoms = bulk("Ar") * [2, 2, 2]
    calc.calculate(atoms)

    assert "energy" in calc.results
    assert "forces" in calc.results
    assert "born_effective_charges" in calc.results

    natoms = len(atoms)
    assert calc.results["born_effective_charges"].shape == (natoms, 3, 3)
    assert calc.results["born_effective_charges"].dtype == np.float32


def test_calculator_bec_get_property():
    model = LoremBEC(cutoff=5.0, num_features=8, num_spherical_features=2, num_radial=4)
    calc = Calculator.from_model(model)

    atoms = bulk("Ar") * [2, 2, 2]
    calc.calculate(atoms)

    bec = calc.get_property("born_effective_charges", atoms)
    assert bec.shape == (len(atoms), 3, 3)


def test_calculator_skin_default():
    """Calculator defaults to skin=0.25."""
    model = Lorem(cutoff=5.0, num_features=8, num_spherical_features=2, num_radial=4)
    calc = Calculator.from_model(model)
    assert calc.skin == 0.25


def test_calculator_skin_configurable():
    """Skin parameter is passed through from_model."""
    model = Lorem(cutoff=5.0, num_features=8, num_spherical_features=2, num_radial=4)
    calc = Calculator.from_model(model, skin=0.5)
    assert calc.skin == 0.5


def test_calculator_skin_results_match_no_skin():
    """Calculator with skin gives same results as skin=0 on static structure."""
    model = Lorem(cutoff=5.0, num_features=8, num_spherical_features=2, num_radial=4)
    atoms = bulk("Ar") * [2, 2, 2]

    calc_skin = Calculator.from_model(model, skin=0.25)
    calc_skin.calculate(atoms)

    calc_no_skin = Calculator.from_model(model, skin=0.0)
    calc_no_skin.calculate(atoms)

    np.testing.assert_allclose(
        calc_skin.results["energy"],
        calc_no_skin.results["energy"],
        atol=1e-5,
    )
    np.testing.assert_allclose(
        calc_skin.results["forces"],
        calc_no_skin.results["forces"],
        atol=1e-5,
    )


def test_calculator_skin_reuses_neighborlist():
    """Small displacement reuses cached neighbor list, results stay correct."""
    model = Lorem(cutoff=5.0, num_features=8, num_spherical_features=2, num_radial=4)
    atoms = bulk("Ar") * [2, 2, 2]

    calc = Calculator.from_model(model, skin=0.5)
    calc.calculate(atoms)

    # Small displacement within skin/2
    displaced = atoms.copy()
    pos = displaced.get_positions()
    pos[0] += 0.05  # 0.087 Å < 0.25 Å = 0.5*skin
    displaced.set_positions(pos)

    # Should not trigger full rebuild (check via cache state)
    assert calc._nl_cache.needs_update(displaced) is False

    calc.calculate(displaced)
    assert "energy" in calc.results
    assert "forces" in calc.results

    # Compare with fresh calculation on displaced structure
    calc_fresh = Calculator.from_model(model, skin=0.5)
    calc_fresh.calculate(displaced)

    np.testing.assert_allclose(
        calc.results["energy"],
        calc_fresh.results["energy"],
        atol=1e-5,
    )
    np.testing.assert_allclose(
        calc.results["forces"],
        calc_fresh.results["forces"],
        atol=1e-5,
    )


def test_calculator_skin_forces_after_displacement():
    """Forces are correct after position-only update (no full rebuild)."""
    model = Lorem(cutoff=5.0, num_features=8, num_spherical_features=2, num_radial=4)
    atoms = bulk("Ar") * [2, 2, 2]
    skin = 0.5

    calc = Calculator.from_model(model, skin=skin)
    calc.calculate(atoms)
    energy_before = calc.results["energy"]

    # Displace and recalculate — should use position-only update
    displaced = atoms.copy()
    pos = displaced.get_positions()
    pos[0, 0] += 0.1
    displaced.set_positions(pos)

    calc.calculate(displaced)
    energy_after = calc.results["energy"]

    # Energy should change (atoms moved)
    assert energy_before != energy_after

    # Forces should have correct shape
    assert calc.results["forces"].shape == (len(atoms), 3)


def test_calculator_skin_full_rebuild_on_large_displacement():
    """Large displacement triggers full neighbor list rebuild."""
    model = Lorem(cutoff=5.0, num_features=8, num_spherical_features=2, num_radial=4)
    atoms = bulk("Ar") * [2, 2, 2]

    calc = Calculator.from_model(model, skin=0.4)
    calc.calculate(atoms)

    # Large displacement exceeding skin/2
    displaced = atoms.copy()
    pos = displaced.get_positions()
    pos[0, 0] += 0.3  # > 0.5 * 0.4 = 0.2 Å
    displaced.set_positions(pos)

    assert calc._nl_cache.needs_update(displaced) is True
    calc.calculate(displaced)
    assert "energy" in calc.results


def test_calculator_cell_change_within_skin():
    """Small cell change (NPT-like) reuses cached neighbor list.

       cell₀                     cell₀ * 1.001
    ┌──────────┐              ┌───────────┐
    │ · · · ·  │  0.1% scale  │ ·  ·  · · │
    │ · · · ·  │  ─────────>  │ ·  ·  · · │  same neighbor
    │ · · · ·  │              │ ·  ·  · · │  list topology
    └──────────┘              └───────────┘
    """
    model = Lorem(cutoff=5.0, num_features=8, num_spherical_features=2, num_radial=4)
    atoms = bulk("Ar") * [2, 2, 2]

    calc = Calculator.from_model(model, skin=0.5)
    calc.calculate(atoms)

    # Small isotropic cell scaling (~0.1%)
    scaled = atoms.copy()
    scaled.set_cell(atoms.get_cell() * 1.001, scale_atoms=True)

    # Should NOT trigger full rebuild
    assert calc._nl_cache.needs_update(scaled) is False

    calc.calculate(scaled)

    # Compare with fresh calculation on scaled structure
    calc_fresh = Calculator.from_model(model, skin=0.5)
    calc_fresh.calculate(scaled)

    np.testing.assert_allclose(
        calc.results["energy"],
        calc_fresh.results["energy"],
        atol=1e-5,
    )
    np.testing.assert_allclose(
        calc.results["forces"],
        calc_fresh.results["forces"],
        atol=1e-5,
    )


def test_calculator_cell_change_stress_correct():
    """Stress is correct after geometry-only update (no full rebuild).

    Stress = ∂E/∂ε depends on sr.cell through:
      σ = Σ_i R_i ⊗ ∂E/∂R_i + Σ_A cell_A ⊗ ∂E/∂cell_A

    Both terms use current sr.positions and sr.cell, so stress
    is correct as long as those are updated.
    """
    model = Lorem(cutoff=5.0, num_features=8, num_spherical_features=2, num_radial=4)
    atoms = bulk("Ar") * [2, 2, 2]

    calc = Calculator.from_model(model, skin=0.5, stress=True)
    calc.calculate(atoms)

    scaled = atoms.copy()
    scaled.set_cell(atoms.get_cell() * 1.002, scale_atoms=True)

    assert calc._nl_cache.needs_update(scaled) is False
    calc.calculate(scaled)

    calc_fresh = Calculator.from_model(model, skin=0.5, stress=True)
    calc_fresh.calculate(scaled)

    np.testing.assert_allclose(
        calc.results["stress"],
        calc_fresh.results["stress"],
        atol=1e-5,
    )


def test_calculator_large_cell_change_triggers_rebuild():
    """Large cell change exceeds combined Verlet criterion."""
    model = Lorem(cutoff=5.0, num_features=8, num_spherical_features=2, num_radial=4)
    atoms = bulk("Ar") * [2, 2, 2]

    calc = Calculator.from_model(model, skin=0.5)
    calc.calculate(atoms)

    scaled = atoms.copy()
    scaled.set_cell(atoms.get_cell() * 1.1, scale_atoms=True)

    assert calc._nl_cache.needs_update(scaled) is True
    calc.calculate(scaled)
    assert "energy" in calc.results


def test_calculator_combined_position_and_cell_change():
    """Both position and cell change within skin — correct results."""
    model = Lorem(cutoff=5.0, num_features=8, num_spherical_features=2, num_radial=4)
    atoms = bulk("Ar") * [2, 2, 2]

    calc = Calculator.from_model(model, skin=0.5)
    calc.calculate(atoms)

    modified = atoms.copy()
    modified.set_cell(atoms.get_cell() * 1.001, scale_atoms=True)
    pos = modified.get_positions()
    pos[0, 0] += 0.05
    modified.set_positions(pos)

    assert calc._nl_cache.needs_update(modified) is False

    calc.calculate(modified)

    calc_fresh = Calculator.from_model(model, skin=0.5)
    calc_fresh.calculate(modified)

    np.testing.assert_allclose(
        calc.results["energy"],
        calc_fresh.results["energy"],
        atol=1e-5,
    )
    np.testing.assert_allclose(
        calc.results["forces"],
        calc_fresh.results["forces"],
        atol=1e-5,
    )


def test_lorem_calculator_import():
    from lorem import LOREMCalculator

    model = LoremBEC(cutoff=5.0, num_features=8, num_spherical_features=2, num_radial=4)
    calc = LOREMCalculator.from_model(model)

    atoms = bulk("Ar") * [2, 2, 2]
    calc.calculate(atoms)

    assert "born_effective_charges" in calc.results
    assert calc.results["born_effective_charges"].shape == (len(atoms), 3, 3)
