import numpy as np
import jax
import jax.numpy as jnp

import e3x
import pytest
from ase.build import bulk, molecule
from ase.calculators.singlepoint import SinglePointCalculator

from lorem.batching import to_batch, to_sample
from lorem.calculator import Calculator
from lorem.models.backbone import ChargeConditioning
from lorem.models.bec import LoremBEC
from lorem.models.mlip import Lorem, LoremQ

# -- data plumbing: atoms.info["total_charge"] -> batch.total_charge --


@pytest.mark.parametrize("charge", [None, -1.0, 2.5])
def test_total_charge_flows_through_batch(charge):
    """batch.total_charge reflects atoms.info["total_charge"], defaulting to
    0.0 when unset."""
    atoms = molecule("H2O")
    if charge is not None:
        atoms.info["total_charge"] = charge
    sample = to_sample(atoms, cutoff=5.0, energy=False, forces=False, stress=False)
    batch = to_batch([sample], [])
    assert float(batch.total_charge[0]) == (0.0 if charge is None else charge)


def test_total_charge_survives_marathon_prepare_roundtrip(tmp_path):
    """total_charge survives marathon.grain.prepare()/DataSource only when
    declared in `properties` -- unlike to_sample()/to_batch() above, which
    always read atoms.info directly, prepare() silently drops undeclared
    entries, and real training datasets go through this path."""
    from marathon.grain import DataSource, prepare

    def make(q):
        atoms = molecule("H2O")
        atoms.info["total_charge"] = q
        atoms.calc = SinglePointCalculator(
            atoms, energy=0.0, forces=np.zeros((len(atoms), 3))
        )
        return atoms

    properties = {
        "energy": {"shape": (1,), "storage": "atoms.calc"},
        "forces": {"shape": ("atom", 3), "storage": "atoms.calc"},
        "total_charge": {"shape": (1,), "storage": "atoms.info"},
    }

    prepare([make(1.0), make(-1.0)], folder=tmp_path / "ds", properties=properties)

    src = DataSource(tmp_path / "ds")
    values = sorted(float(src[i].info["total_charge"]) for i in range(len(src)))
    assert values == [-1.0, 1.0]


def test_missing_total_charge_warns_once(capsys):
    """A missing atoms.info["total_charge"] warns exactly once, even across
    repeated calls."""
    import lorem.batching as batching

    batching._warned_missing_total_charge = False
    try:
        for _ in range(3):
            to_sample(molecule("H2O"), cutoff=5.0, energy=False, forces=False, stress=False)
    finally:
        batching._warned_missing_total_charge = False

    out = capsys.readouterr().out
    assert out.count("not set; assuming") == 1


# -- ChargeConditioning --


def test_charge_conditioning_module_changes_with_Q():
    """The ChargeConditioning FiLM layer's output depends on the per-atom
    charge Q_i."""
    key = jax.random.key(0)
    num_atoms, d = 4, 6
    x = jax.random.normal(key, (num_atoms, d))
    atom_mask = jnp.ones(num_atoms, dtype=bool)
    Q_i = jnp.array([1.0, 1.0, -1.0, -1.0])

    model = ChargeConditioning(features=d)
    params = model.init(key, Q_i, x, atom_mask)
    y = model.apply(params, Q_i, x, atom_mask)
    y_zero_Q = model.apply(params, jnp.zeros(num_atoms), x, atom_mask)

    assert y.shape == (num_atoms, d)
    assert not jnp.allclose(y, y_zero_Q)


# -- end-to-end: Lorem/LoremBEC on hand-built structures --


# max_degree defaults to 6 on the real model: 49 lm components and a
# 343-path CG kernel, which dominates XLA compile time and is recompiled for
# every distinct model in this file. These tests check plumbing and derivative
# correctness, neither of which is degree-specific, so they run at 2. The
# rotation test below overrides it back to 6, since that is where a
# degree-specific equivariance bug would actually show up.
TEST_MAX_DEGREE = 2


def _make_model(lr=False, max_degree=TEST_MAX_DEGREE):
    return Lorem(
        cutoff=4.0,
        max_degree=max_degree,
        num_features=8,
        num_spherical_features=2,
        num_radial=4,
        num_message_passing=1,
        lr=lr,
    )


def test_bec_charge_conditioning_differs_with_Q():
    """LoremBEC predicts different energy for a structure at Q=+1 vs Q=-1
    (no other test exercises charge conditioning on LoremBEC)."""
    model = LoremBEC(
        cutoff=4.0,
        max_degree=TEST_MAX_DEGREE,
        num_features=8,
        num_spherical_features=2,
        num_radial=4,
    )
    atoms = bulk("Ar") * [2, 2, 2]

    atoms_plus = atoms.copy()
    atoms_plus.info["total_charge"] = 1.0
    calc_plus = Calculator.from_model(model)
    calc_plus.calculate(atoms_plus)

    atoms_minus = atoms.copy()
    atoms_minus.info["total_charge"] = -1.0
    calc_minus = Calculator.from_model(model)
    calc_minus.calculate(atoms_minus)

    assert not np.allclose(
        calc_plus.results["energy"], calc_minus.results["energy"], atol=1e-6
    )


def test_water_smoke():
    """Energy/forces stay finite for a hand-built water molecule across
    charge states, using total_charge exactly as it flows through
    atoms.info -> to_sample -> to_batch."""
    model = _make_model()
    calc = Calculator.from_model(model)
    for q in (-1.0, 0.0, 1.0):
        atoms = molecule("H2O")
        atoms.info["total_charge"] = q
        calc.calculate(atoms)
        assert np.all(np.isfinite(calc.results["energy"]))
        assert np.all(np.isfinite(calc.results["forces"]))


# -- Calculator cache invalidation: total_charge lives in atoms.info, not
# positions/cell, so a reused Calculator instance must detect changes to it
# independently of the neighbor-list/geometry cache (e.g. a charge sweep at
# fixed geometry) --


def test_calculator_picks_up_total_charge_change_at_fixed_geometry():
    """Reusing one Calculator across a total_charge change at fixed geometry
    recomputes rather than caching stale results, and matches a fresh
    Calculator's output -- this also covers plain Lorem's basic
    charge-sensitivity (e_plus != e_minus), so no separate test for that
    is needed."""
    model = _make_model()
    key = jax.random.key(0)
    params = model.init(key, *model.dummy_inputs())

    calc = Calculator.from_model(model, params=params)
    atoms = molecule("H2O")

    atoms.info["total_charge"] = 1.0
    calc.calculate(atoms)
    e_plus = calc.results["energy"]

    atoms.info["total_charge"] = -1.0
    calc.calculate(atoms)
    e_minus = calc.results["energy"]

    assert not np.allclose(e_plus, e_minus, atol=1e-6)

    fresh_calc = Calculator.from_model(model, params=params)
    atoms_minus = molecule("H2O")
    atoms_minus.info["total_charge"] = -1.0
    fresh_calc.calculate(atoms_minus)

    assert np.allclose(e_minus, fresh_calc.results["energy"], atol=1e-6)


# -- work function (dE/dq) as a prediction target --
#
# predict() takes value_and_grad of the energy w.r.t. the whole batch, so the
# derivative w.r.t. the per-structure total_charge vector comes out of the same
# backward pass as the forces. These pin down that it is really dE/dq, that
# padded structures stay zero, and that the extra key is inert when nothing
# asks for it.


def _make_q_model(predict_bec=False, max_degree=TEST_MAX_DEGREE):
    return LoremQ(
        cutoff=4.0,
        max_degree=max_degree,
        num_features=8,
        num_spherical_features=2,
        num_radial=4,
        num_message_passing=1,
        predict_bec=predict_bec,
    )


def _batch_at_charge(model, q):
    atoms = molecule("H2O")
    atoms.info["total_charge"] = q
    return model.atoms_to_batch(atoms)


def test_work_function_matches_central_difference():
    model = _make_q_model()
    key = jax.random.key(0)
    params = model.init(key, *model.dummy_inputs())

    # h is near-optimal for a float32 central difference: the truncation error
    # is O(h^2) and the roundoff floor is O(eps/h), which balance at
    # h ~ eps^(1/3) ~ 5e-3. Tolerance is set to that floor (~1e-3 relative),
    # not to autodiff precision -- the finite difference is the inaccurate side
    # of this comparison, not the gradient.
    q0, h = 0.3, 5e-3

    wf = model.predict(params, _batch_at_charge(model, q0))["work_function"][0]

    e_plus, _ = model.energy(params, _batch_at_charge(model, q0 + h))
    e_minus, _ = model.energy(params, _batch_at_charge(model, q0 - h))
    finite_difference = (e_plus - e_minus) / (2.0 * h)

    np.testing.assert_allclose(wf, finite_difference, rtol=1e-3, atol=1e-5)


def test_work_function_is_zero_on_padded_structures():
    model = _make_q_model()
    key = jax.random.key(0)
    params = model.init(key, *model.dummy_inputs())

    batch = _batch_at_charge(model, 0.5)
    results = model.predict(params, batch)

    # to_batch pads to a power of 2, so slot 1 is padding
    assert not bool(batch.sr.structure_mask[1])
    assert float(results["work_function"][1]) == 0.0
    assert results["work_function"].shape == results["energy"].shape


def test_work_function_varies_with_charge():
    """A constant dE/dq would still pass the finite-difference test at a single
    point; this catches a readout that ignores Q beyond a linear term."""
    model = _make_q_model()
    key = jax.random.key(0)
    params = model.init(key, *model.dummy_inputs())

    values = [
        float(model.predict(params, _batch_at_charge(model, q))["work_function"][0])
        for q in (-1.0, 0.0, 1.0)
    ]
    assert len(set(values)) == 3


def test_work_function_is_rotation_invariant():
    R = np.array(e3x.so3.random_rotation(jax.random.key(0)))
    model = _make_q_model(max_degree=6)
    key = jax.random.key(0)
    params = model.init(key, *model.dummy_inputs())

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

    model = _make_q_model()
    key = jax.random.key(0)
    params = model.init(key, *model.dummy_inputs())

    atoms = molecule("H2O")
    atoms.info["total_charge"] = 0.4
    atoms.calc = SinglePointCalculator(
        atoms, energy=-1.0, forces=np.zeros((len(atoms), 3))
    )
    sample = to_sample(atoms, cutoff=model.cutoff, keys=["energy", "forces"])
    batch = jax.tree.map(jnp.asarray, to_batch([sample], ["energy", "forces"]))

    assert "work_function" not in batch.labels

    loss_fn = get_loss_fn(
        lambda p, b: model.predict(p, b), weights={"energy": 0.5, "forces": 0.5}
    )
    loss, aux = loss_fn(params, batch)

    assert np.isfinite(float(loss))
    assert not any(k.startswith("work_function") for k in aux)


# -- Born effective charges (d2E/dr dq) --


def test_bec_z_matches_finite_difference_of_forces():
    """Z* = (A eps0) dF/dq. Check the autograd mixed derivative against a
    central difference of the forces in q, with the same float32-aware
    tolerance rationale as the work-function test."""
    from lorem.models.mlip import EPSILON_0

    model = _make_q_model(predict_bec=True)
    key = jax.random.key(0)
    params = model.init(key, *model.dummy_inputs())

    q0, h = 0.3, 5e-3
    bec = model.predict(params, _batch_at_charge(model, q0))["bec_z"]

    f_plus = model.predict(params, _batch_at_charge(model, q0 + h))["forces"]
    f_minus = model.predict(params, _batch_at_charge(model, q0 - h))["forces"]
    dFdq = (f_plus - f_minus) / (2.0 * h)

    batch = _batch_at_charge(model, q0)
    cell = batch.sr.cell
    area = float(np.linalg.norm(np.cross(cell[0, 0], cell[0, 1])))
    expected = dFdq * area * EPSILON_0

    mask = np.asarray(batch.sr.atom_mask)
    np.testing.assert_allclose(
        np.asarray(bec)[mask], np.asarray(expected)[mask], rtol=2e-2, atol=1e-4
    )


def test_bec_z_absent_unless_requested():
    model = _make_q_model(predict_bec=False)
    key = jax.random.key(0)
    params = model.init(key, *model.dummy_inputs())
    assert "bec_z" not in model.predict(params, _batch_at_charge(model, 0.3))


def test_bec_z_is_zero_on_padded_atoms():
    model = _make_q_model(predict_bec=True)
    key = jax.random.key(0)
    params = model.init(key, *model.dummy_inputs())
    batch = _batch_at_charge(model, 0.3)
    bec = np.asarray(model.predict(params, batch)["bec_z"])
    pad = ~np.asarray(batch.sr.atom_mask)
    assert bec.shape == np.asarray(batch.sr.positions).shape
    if pad.any():
        assert np.allclose(bec[pad], 0.0)


def test_plain_lorem_reports_no_charge_derivatives():
    """The point of the Lorem/LoremQ split: a plain MLIP must not hand back an
    unconstrained dE/dq alongside energy and forces. Lorem applies
    ChargeConditioning unconditionally and total_charge defaults to 0, so any
    work_function it reported would be pure extrapolation."""
    model = _make_model()
    key = jax.random.key(0)
    params = model.init(key, *model.dummy_inputs())
    results = model.predict(params, _batch_at_charge(model, 0.3))

    assert set(results) == {"energy", "forces"}
    assert "work_function" not in results
    assert "bec_z" not in results

