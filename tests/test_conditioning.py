import numpy as np
import jax
import jax.numpy as jnp

import pytest
from ase.build import bulk, molecule
from marathon.data.properties import DEFAULT_PROPERTIES

from lorem.calculator import Calculator
from lorem.models.backbone import ChargeConditioning
from lorem.models.bec import LoremBEC
from lorem.models.mlip import Lorem

PROPERTIES = {
    **DEFAULT_PROPERTIES,
    "total_charge": {"shape": (1,), "storage": "atoms.info"},
}


def test_charge_conditioning_module():
    key = jax.random.key(0)
    num_atoms, d = 4, 6
    x = jax.random.normal(key, (num_atoms, d))
    atom_mask = jnp.ones(num_atoms, dtype=bool)
    Q_i = jnp.array([1.0, 1.0, -1.0, -1.0])

    model = ChargeConditioning(features=d)
    params = model.init(key, Q_i, x, atom_mask)

    assert not jnp.allclose(model.apply(params, Q_i, x, atom_mask), x)
    np.testing.assert_allclose(model.apply(params, jnp.zeros(num_atoms), x, atom_mask), x)


@pytest.mark.parametrize("cls", [Lorem, LoremBEC])
def test_energy_depends_on_total_charge(cls):
    model = cls(
        cutoff=5.0,
        num_features=8,
        num_spherical_features=2,
        num_radial=4,
        num_message_passing=1,
        charge_conditioning=True,
    )
    calc = Calculator.from_model(model, properties=PROPERTIES)

    energies = []
    for atoms, q in [
        (molecule("H2O"), -1.0),
        (molecule("H2O"), 1.0),
        (bulk("Ar") * 2, 1.0),
    ]:
        atoms.info["total_charge"] = q
        calc.calculate(atoms)
        assert np.all(np.isfinite(calc.results["energy"]))
        assert np.all(np.isfinite(calc.results["forces"]))
        energies.append(calc.results["energy"])

    assert not np.allclose(energies[0], energies[1], atol=1e-6)
