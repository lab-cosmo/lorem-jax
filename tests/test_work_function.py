import numpy as np
import jax
import jax.numpy as jnp

import e3x
import pytest
from ase.build import bulk, molecule
from ase.calculators.singlepoint import SinglePointCalculator

from lorem.batching import to_batch, to_sample
from lorem.calculator import Calculator
from lorem.models.work_function import LoremWF

# 2, not the production 6: max_degree drives a 343-path CG kernel that
# dominates XLA compile time, and nothing here is degree-specific except the
# rotation test, which overrides it back.
TEST_MAX_DEGREE = 2

HEADS = [True, False]


def _make_model(from_energy=True, max_degree=TEST_MAX_DEGREE, cutoff=4.0, **kwargs):
    return LoremWF(
        cutoff=cutoff,
        max_degree=max_degree,
        num_features=8,
        num_spherical_features=2,
        num_radial=4,
        num_message_passing=1,
        lr=False,
        work_function_from_energy=from_energy,
        **kwargs,
    )


def _batch_at_charge(model, q):
    atoms = molecule("H2O")
    atoms.info["total_charge"] = q
    return model.atoms_to_batch(atoms)


def _init(model, seed=0):
    return model.init(jax.random.key(seed), *model.dummy_inputs())


# -- work_function_from_energy=True --


def test_work_function_matches_central_difference():
    """h=5e-3 balances the O(h^2) truncation error against the O(eps/h)
    float32 roundoff floor, and the tolerance is that floor -- the finite
    difference is the inaccurate side of this comparison, not the gradient."""
    model = _make_model()
    params = _init(model)

    q0, h = 0.3, 5e-3

    wf = model.predict(params, _batch_at_charge(model, q0))["work_function"][0]

    e_plus, _ = model.energy(params, _batch_at_charge(model, q0 + h))
    e_minus, _ = model.energy(params, _batch_at_charge(model, q0 - h))

    np.testing.assert_allclose(wf, (e_plus - e_minus) / (2.0 * h), rtol=1e-3, atol=1e-5)


def test_work_function_varies_with_charge():
    """A constant dE/dq would still pass the finite-difference test at a
    single point."""
    model = _make_model()
    params = _init(model)

    values = [
        float(model.predict(params, _batch_at_charge(model, q))["work_function"][0])
        for q in (-1.0, 0.0, 1.0)
    ]
    assert len(set(values)) == 3


# -- both heads --


@pytest.mark.parametrize("from_energy", HEADS)
def test_work_function_is_zero_on_padded_structures(from_energy):
    model = _make_model(from_energy)
    params = _init(model)

    batch = _batch_at_charge(model, 0.5)
    results = model.predict(params, batch)

    assert not bool(batch.sr.structure_mask[1])  # to_batch pads to a power of 2
    assert float(results["work_function"][1]) == 0.0
    assert results["work_function"].shape == results["energy"].shape


@pytest.mark.parametrize("from_energy", HEADS)
def test_work_function_is_rotation_invariant(from_energy):
    R = np.array(e3x.so3.random_rotation(jax.random.key(0)))
    model = _make_model(from_energy, max_degree=6)
    params = _init(model)

    atoms = molecule("H2O")
    atoms.info["total_charge"] = 0.7
    wf = model.predict(params, model.atoms_to_batch(atoms))["work_function"][0]

    atoms_rot = atoms.copy()
    atoms_rot.positions = atoms.positions @ R.T
    wf_rot = model.predict(params, model.atoms_to_batch(atoms_rot))["work_function"][0]

    np.testing.assert_allclose(wf, wf_rot, atol=1e-4)


def test_work_function_key_is_inert_without_a_label():
    """predict() always returns work_function, so adding the key must not
    change runs that don't train on it."""
    from marathon.evaluate.loss import get_loss_fn

    model = _make_model()
    params = _init(model)

    atoms = molecule("H2O")
    atoms.info["total_charge"] = 0.4
    atoms.calc = SinglePointCalculator(atoms, energy=-1.0, forces=np.zeros((len(atoms), 3)))
    sample = to_sample(atoms, cutoff=model.cutoff, keys=["energy", "forces"])
    batch = jax.tree.map(jnp.asarray, to_batch([sample], ["energy", "forces"]))

    assert "work_function" not in batch.labels

    loss_fn = get_loss_fn(
        lambda p, b: model.predict(p, b), weights={"energy": 0.5, "forces": 0.5}
    )
    loss, aux = loss_fn(params, batch)

    assert np.isfinite(float(loss))
    assert not any(k.startswith("work_function") for k in aux)


# -- work_function_from_energy=False --


def test_direct_head_is_intensive():
    """The head must mean-pool where the energy readout sums."""
    model = _make_model(from_energy=False, cutoff=3.0)
    params = _init(model)

    atoms = bulk("Ar", cubic=True) * [2, 2, 2]
    atoms.info["total_charge"] = 0.5
    doubled = atoms * [2, 1, 1]
    doubled.info["total_charge"] = 0.5

    single = model.predict(params, model.atoms_to_batch(atoms))
    double = model.predict(params, model.atoms_to_batch(doubled))

    np.testing.assert_allclose(double["energy"][0], 2.0 * single["energy"][0], rtol=1e-4)
    np.testing.assert_allclose(
        double["work_function"][0], single["work_function"][0], rtol=1e-4
    )


def test_direct_head_does_not_change_energy_or_forces():
    """Only jnp.sum(energies) is differentiated, so the head is a pure
    side-output."""
    model = _make_model(from_energy=False)
    params = _init(model)
    batch = _batch_at_charge(model, 0.3)
    before = model.predict(params, batch)

    perturbed = jax.tree.map(lambda x: x + 1.0, params["params"]["PooledScalarHead_0"])
    params = {"params": {**params["params"], "PooledScalarHead_0": perturbed}}
    after = model.predict(params, batch)

    np.testing.assert_array_equal(after["energy"], before["energy"])
    np.testing.assert_array_equal(after["forces"], before["forces"])
    assert not np.allclose(after["work_function"], before["work_function"])


# -- the ASE path --


@pytest.mark.parametrize("from_energy", HEADS)
def test_calculator_exposes_work_function(from_energy):
    model = _make_model(from_energy)
    calc = Calculator.from_model(model)

    atoms = molecule("H2O")
    atoms.info["total_charge"] = 0.5
    calc.calculate(atoms)

    assert "work_function" in calc.implemented_properties
    assert np.isfinite(calc.results["work_function"])
    assert isinstance(calc.get_property("work_function", atoms), float)


def test_calculator_work_function_tracks_a_charge_sweep():
    """total_charge lives in atoms.info, invisible to the neighbor-list and
    geometry caches."""
    model = _make_model()
    params = _init(model)
    calc = Calculator.from_model(model, params=params)

    atoms = molecule("H2O")
    values = []
    for q in (-1.0, 0.0, 1.0):
        atoms.info["total_charge"] = q
        calc.calculate(atoms)
        values.append(calc.results["work_function"])

    assert len(set(values)) == 3


def test_plain_lorem_reports_no_work_function():
    """A plain MLIP must not hand back an unconstrained dE/dq alongside
    energy and forces."""
    from lorem.models.mlip import Lorem

    model = Lorem(
        cutoff=4.0,
        max_degree=TEST_MAX_DEGREE,
        num_features=8,
        num_spherical_features=2,
        num_radial=4,
        lr=False,
    )
    params = _init(model)

    assert set(model.predict(params, _batch_at_charge(model, 0.3))) == {
        "energy",
        "forces",
    }


def test_offset_defaults_to_none_and_acts_as_zero():
    """`None` means "take the value prepare.py fitted", which a bare model has
    no dataset to look up -- so it must behave as no offset rather than fail."""
    model = _make_model(from_energy=False)
    assert model.work_function_offset is None

    params = _init(model)
    batch = _batch_at_charge(model, 0.3)
    unset = model.predict(params, batch)["work_function"]

    explicit = _make_model(from_energy=False, work_function_offset=0.0)
    zero = explicit.predict(_init(explicit), batch)["work_function"]

    np.testing.assert_allclose(unset, zero, rtol=1e-6)


def test_offset_shifts_only_real_structures():
    """A resolved offset is a plain additive shift, and must stay off the
    padding -- the masking runs after it is added."""
    model = _make_model(from_energy=False, work_function_offset=4.5)
    params = _init(model)
    batch = _batch_at_charge(model, 0.3)
    shifted = model.predict(params, batch)["work_function"]

    base = _make_model(from_energy=False, work_function_offset=0.0)
    plain = base.predict(_init(base), batch)["work_function"]

    np.testing.assert_allclose(shifted[0], plain[0] + 4.5, rtol=1e-5)
    assert float(shifted[1]) == 0.0
