import numpy as np
import jax
import jax.numpy as jnp

import e3x
import pytest
from ase.build import bulk, molecule
from ase.calculators.singlepoint import SinglePointCalculator

from lorem.batching import to_batch, to_sample
from lorem.calculator import Calculator
from lorem.models.loremq import LoremQ

# max_degree defaults to 6 on the real model: 49 lm components and a 343-path
# CG kernel, which dominates XLA compile time and is recompiled for every
# distinct model here. These tests check plumbing and derivative correctness,
# neither of which is degree-specific, so they run at 2. The rotation test
# overrides it back to 6, since that is where a degree-specific bug shows up.
TEST_MAX_DEGREE = 2

HEADS = ["autodiff", "direct"]


def _make_model(head="autodiff", max_degree=TEST_MAX_DEGREE, cutoff=4.0, **kwargs):
    return LoremQ(
        cutoff=cutoff,
        max_degree=max_degree,
        num_features=8,
        num_spherical_features=2,
        num_radial=4,
        num_message_passing=1,
        lr=False,
        work_function_head=head,
        **kwargs,
    )


def _batch_at_charge(model, q):
    atoms = molecule("H2O")
    atoms.info["total_charge"] = q
    return model.atoms_to_batch(atoms)


def _init(model, seed=0):
    return model.init(jax.random.key(seed), *model.dummy_inputs())


# -- the autodiff head: is it really dE/dq? --


def test_autodiff_work_function_matches_central_difference():
    """h is near-optimal for a float32 central difference: truncation error is
    O(h^2) and the roundoff floor is O(eps/h), balancing at h ~ eps^(1/3) ~
    5e-3. The tolerance is that floor, not autodiff precision -- the finite
    difference is the inaccurate side of this comparison."""
    model = _make_model("autodiff")
    params = _init(model)

    q0, h = 0.3, 5e-3

    wf = model.predict(params, _batch_at_charge(model, q0))["work_function"][0]

    e_plus, _ = model.energy(params, _batch_at_charge(model, q0 + h))
    e_minus, _ = model.energy(params, _batch_at_charge(model, q0 - h))
    finite_difference = (e_plus - e_minus) / (2.0 * h)

    np.testing.assert_allclose(wf, finite_difference, rtol=1e-3, atol=1e-5)


def test_autodiff_work_function_varies_with_charge():
    """A constant dE/dq would still pass the finite-difference test at a single
    point; this catches an energy that is merely linear in Q."""
    model = _make_model("autodiff")
    params = _init(model)

    values = [
        float(model.predict(params, _batch_at_charge(model, q))["work_function"][0])
        for q in (-1.0, 0.0, 1.0)
    ]
    assert len(set(values)) == 3


# -- shared contract of both heads --


@pytest.mark.parametrize("head", HEADS)
def test_work_function_is_zero_on_padded_structures(head):
    model = _make_model(head)
    params = _init(model)

    batch = _batch_at_charge(model, 0.5)
    results = model.predict(params, batch)

    # to_batch pads to a power of 2, so slot 1 is padding
    assert not bool(batch.sr.structure_mask[1])
    assert float(results["work_function"][1]) == 0.0
    assert results["work_function"].shape == results["energy"].shape


@pytest.mark.parametrize("head", HEADS)
def test_work_function_is_rotation_invariant(head):
    R = np.array(e3x.so3.random_rotation(jax.random.key(0)))
    model = _make_model(head, max_degree=6)
    params = _init(model)

    atoms = molecule("H2O")
    atoms.info["total_charge"] = 0.7
    wf = model.predict(params, model.atoms_to_batch(atoms))["work_function"][0]

    atoms_rot = atoms.copy()
    atoms_rot.positions = atoms.positions @ R.T
    wf_rot = model.predict(params, model.atoms_to_batch(atoms_rot))["work_function"][0]

    np.testing.assert_allclose(wf, wf_rot, atol=1e-4)


def test_work_function_key_is_inert_without_a_label():
    """predict() always returns work_function, but the loss must ignore it
    unless it is in loss_weights -- otherwise adding the key would silently
    change runs that don't train on it."""
    from marathon.evaluate.loss import get_loss_fn

    model = _make_model("autodiff")
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


def test_unknown_work_function_head_raises():
    model = _make_model("pooled")
    with pytest.raises(ValueError, match="unknown work_function_head"):
        _init(model)


# -- the direct head --


def test_direct_head_is_intensive():
    """The work function is intensive, so the head must mean-pool where the
    energy readout sums. A doubled cell has twice the energy and the same
    work function."""
    model = _make_model("direct", cutoff=3.0)
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
    side-output: perturbing its weights must move nothing else."""
    model = _make_model("direct")
    params = _init(model)
    batch = _batch_at_charge(model, 0.3)
    before = model.predict(params, batch)

    perturbed = jax.tree.map(lambda x: x + 1.0, params["params"]["PooledScalarHead_0"])
    params = {"params": {**params["params"], "PooledScalarHead_0": perturbed}}
    after = model.predict(params, batch)

    np.testing.assert_array_equal(after["energy"], before["energy"])
    np.testing.assert_array_equal(after["forces"], before["forces"])
    assert not np.allclose(after["work_function"], before["work_function"])


def test_work_function_offset_shifts_the_direct_head():
    """The offset starts the head near the label mean; it must be a plain
    additive shift on real structures and stay off the padding."""
    offset = 4.5
    batch = None
    values = {}
    for value in (0.0, offset):
        model = _make_model("direct", work_function_offset=value)
        params = _init(model)
        batch = _batch_at_charge(model, 0.3)
        values[value] = model.predict(params, batch)["work_function"]

    np.testing.assert_allclose(values[offset][0], values[0.0][0] + offset, rtol=1e-5)
    assert float(values[offset][1]) == 0.0


# -- the ASE path --


@pytest.mark.parametrize("head", HEADS)
def test_calculator_exposes_work_function(head):
    model = _make_model(head)
    calc = Calculator.from_model(model)

    atoms = molecule("H2O")
    atoms.info["total_charge"] = 0.5
    calc.calculate(atoms)

    assert "work_function" in calc.implemented_properties
    assert np.isfinite(calc.results["work_function"])
    assert isinstance(calc.get_property("work_function", atoms), float)


def test_calculator_work_function_tracks_a_charge_sweep():
    """total_charge lives in atoms.info, invisible to the neighbor-list and
    geometry caches, so a reused Calculator must recompute the work function
    when only the charge changes."""
    model = _make_model("autodiff")
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
    """The Lorem/LoremQ split: a plain MLIP must not hand back an
    unconstrained dE/dq alongside energy and forces."""
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
    results = model.predict(params, _batch_at_charge(model, 0.3))

    assert set(results) == {"energy", "forces"}
