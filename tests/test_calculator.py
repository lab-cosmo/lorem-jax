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
    assert "BEC" in calc.results

    natoms = len(atoms)
    assert calc.results["BEC"].shape == (3 * natoms, 3)
    assert calc.results["BEC"].dtype == np.float32


def test_calculator_bec_get_property():
    model = LoremBEC(cutoff=5.0, num_features=8, num_spherical_features=2, num_radial=4)
    calc = Calculator.from_model(model)

    atoms = bulk("Ar") * [2, 2, 2]
    calc.calculate(atoms)

    bec = calc.get_property("BEC", atoms)
    assert bec.shape == (3 * len(atoms), 3)


def test_lorem_calculator_import():
    from lorem import LOREMCalculator

    model = LoremBEC(cutoff=5.0, num_features=8, num_spherical_features=2, num_radial=4)
    calc = LOREMCalculator.from_model(model)

    atoms = bulk("Ar") * [2, 2, 2]
    calc.calculate(atoms)

    assert "BEC" in calc.results
    assert calc.results["BEC"].shape == (3 * len(atoms), 3)
