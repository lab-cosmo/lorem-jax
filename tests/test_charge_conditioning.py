import numpy as np
import jax
import jax.numpy as jnp

from pathlib import Path

from ase.build import bulk, molecule
from ase.io import read

from lorem.batching import to_batch, to_sample
from lorem.calculator import Calculator
from lorem.models.backbone import (
    ChargeFiLM,
    ChargeInit,
    ChargeUpdate,
    conserving_redistribution,
    segment_softmax,
)
from lorem.models.bec import LoremBEC
from lorem.models.mlip import Lorem

AG_CLUSTERS = Path(__file__).parent / "data" / "ag_clusters_small.xyz"


# -- data plumbing: atoms.info["total_charge"] -> batch.total_charge --


def test_total_charge_flows_through_batch():
    atoms = molecule("H2O")
    atoms.info["total_charge"] = -1.0
    sample = to_sample(atoms, cutoff=5.0, energy=False, forces=False, stress=False)
    batch = to_batch([sample], [])
    assert float(batch.total_charge[0]) == -1.0


def test_total_charge_defaults_to_zero():
    atoms = molecule("H2O")
    sample = to_sample(atoms, cutoff=5.0, energy=False, forces=False, stress=False)
    batch = to_batch([sample], [])
    assert float(batch.total_charge[0]) == 0.0


def test_missing_total_charge_warns_once(capsys):
    import lorem.batching as batching

    batching._warned_missing_total_charge = False
    try:
        for _ in range(3):
            to_sample(molecule("H2O"), cutoff=5.0, energy=False, forces=False, stress=False)
    finally:
        batching._warned_missing_total_charge = False

    out = capsys.readouterr().out
    assert out.count("not set; assuming") == 1


# -- segment_softmax --


def test_segment_softmax_sums_to_one_per_segment():
    logits = jnp.array([1.0, 2.0, 0.5, 3.0])
    segment_ids = jnp.array([0, 0, 1, 1])
    mask = jnp.ones(4, dtype=bool)
    weights = segment_softmax(logits, segment_ids, mask, num_segments=2)
    sums = jax.ops.segment_sum(weights, segment_ids, num_segments=2)
    np.testing.assert_allclose(sums, [1.0, 1.0], atol=1e-6)


def test_segment_softmax_respects_mask():
    logits = jnp.array([1.0, 2.0, 0.5])
    segment_ids = jnp.array([0, 0, 0])
    mask = jnp.array([True, True, False])
    weights = segment_softmax(logits, segment_ids, mask, num_segments=1)
    assert float(weights[2]) == 0.0
    np.testing.assert_allclose(weights[0] + weights[1], 1.0, atol=1e-6)


def test_segment_softmax_empty_segment_no_nan():
    logits = jnp.array([1.0, 2.0])
    segment_ids = jnp.array([0, 0])
    mask = jnp.array([True, True])
    # num_segments=2 but nothing maps to segment 1 (padding structure)
    weights = segment_softmax(logits, segment_ids, mask, num_segments=2)
    assert not bool(jnp.any(jnp.isnan(weights)))
    sums = jax.ops.segment_sum(weights, segment_ids, num_segments=2)
    np.testing.assert_allclose(sums, [1.0, 0.0], atol=1e-6)


# -- conserving_redistribution --


def test_conserving_redistribution_matches_target():
    values = jnp.array([0.1, 0.2, 0.3, 0.4])
    segment_ids = jnp.array([0, 0, 1, 1])
    weights = jnp.array([1.0, 1.0, 2.0, 1.0])
    mask = jnp.ones(4, dtype=bool)
    target = jnp.array([1.0, -1.0])
    result = conserving_redistribution(values, target, weights, segment_ids, mask, 2)
    sums = jax.ops.segment_sum(result, segment_ids, num_segments=2)
    np.testing.assert_allclose(sums, target, atol=1e-6)


def test_conserving_redistribution_respects_mask():
    values = jnp.array([0.1, 0.2, 0.3])
    segment_ids = jnp.array([0, 0, 0])
    weights = jnp.array([1.0, 1.0, 5.0])
    mask = jnp.array([True, True, False])
    target = jnp.array([2.0])
    result = conserving_redistribution(values, target, weights, segment_ids, mask, 1)
    assert float(result[2]) == 0.0
    np.testing.assert_allclose(result[0] + result[1], 2.0, atol=1e-6)


# -- ChargeFiLM / ChargeInit / ChargeUpdate modules --


def test_charge_film_changes_with_Q():
    key = jax.random.key(0)
    num_atoms, d = 4, 6
    x = jax.random.normal(key, (num_atoms, d))
    atom_mask = jnp.ones(num_atoms, dtype=bool)
    Q_i = jnp.array([1.0, 1.0, -1.0, -1.0])

    model = ChargeFiLM(features=d)
    params = model.init(key, Q_i, x, atom_mask)
    y = model.apply(params, Q_i, x, atom_mask)
    y_zero_Q = model.apply(params, jnp.zeros(num_atoms), x, atom_mask)

    assert y.shape == (num_atoms, d)
    assert not jnp.allclose(y, y_zero_Q)


def test_charge_init_sums_to_Q():
    key = jax.random.key(0)
    num_atoms, d = 5, 4
    h = jax.random.normal(key, (num_atoms, d))
    Q = jnp.array([1.0, -1.0])
    atom_to_structure = jnp.array([0, 0, 0, 1, 1])
    atom_mask = jnp.ones(num_atoms, dtype=bool)

    model = ChargeInit(features=d)
    params = model.init(key, h, Q, atom_to_structure, atom_mask, 2)
    c = model.apply(params, h, Q, atom_to_structure, atom_mask, 2)

    sums = jax.ops.segment_sum(c, atom_to_structure, num_segments=2)
    np.testing.assert_allclose(sums, Q, atol=1e-5)


def test_charge_init_respects_padding():
    key = jax.random.key(0)
    num_atoms, d = 4, 4
    h = jax.random.normal(key, (num_atoms, d))
    Q = jnp.array([1.0, 0.0])  # second "structure" is padding
    atom_to_structure = jnp.array([0, 0, 0, 1])
    atom_mask = jnp.array([True, True, True, False])

    model = ChargeInit(features=d)
    params = model.init(key, h, Q, atom_to_structure, atom_mask, 2)
    c = model.apply(params, h, Q, atom_to_structure, atom_mask, 2)

    assert float(c[3]) == 0.0
    np.testing.assert_allclose(jnp.sum(c[:3]), 1.0, atol=1e-5)


def test_charge_update_preserves_conservation():
    key = jax.random.key(1)
    num_atoms, d = 5, 4
    h = jax.random.normal(key, (num_atoms, d))
    Q = jnp.array([1.0, -1.0])
    atom_to_structure = jnp.array([0, 0, 0, 1, 1])
    atom_mask = jnp.ones(num_atoms, dtype=bool)

    init_model = ChargeInit(features=d)
    init_params = init_model.init(key, h, Q, atom_to_structure, atom_mask, 2)
    c = init_model.apply(init_params, h, Q, atom_to_structure, atom_mask, 2)

    # a small graph with edges only within each structure
    i = jnp.array([0, 1, 1, 2, 3, 4])
    edges_scalar = jax.random.normal(jax.random.key(2), (6, d))
    pair_mask = jnp.ones(6, dtype=bool)

    update_model = ChargeUpdate(features=d)
    args = (c, h, edges_scalar, i, pair_mask, atom_mask, Q, atom_to_structure, num_atoms, 2)
    update_params = update_model.init(jax.random.key(3), *args)
    c_new = update_model.apply(update_params, *args)

    sums = jax.ops.segment_sum(c_new, atom_to_structure, num_segments=2)
    np.testing.assert_allclose(sums, Q, atol=1e-5)


# -- end-to-end: Lorem model with the Ag3+/Ag3- cluster data --


def _make_model(charge_conditioning):
    return Lorem(
        cutoff=6.0,
        num_features=8,
        num_spherical_features=2,
        num_radial=4,
        num_message_passing=1,
        lr=False,
        charge_conditioning=charge_conditioning,
    )


def test_charge_conditioning_none_ignores_Q():
    atoms = read(AG_CLUSTERS, index=0)

    model = _make_model("none")

    atoms_plus = atoms.copy()
    atoms_plus.info["total_charge"] = 1.0
    calc_plus = Calculator.from_model(model)
    calc_plus.calculate(atoms_plus)

    atoms_minus = atoms.copy()
    atoms_minus.info["total_charge"] = -1.0
    calc_minus = Calculator.from_model(model)
    calc_minus.calculate(atoms_minus)

    np.testing.assert_allclose(
        calc_plus.results["energy"], calc_minus.results["energy"], atol=1e-6
    )


def test_charge_conditioning_film_differs_with_Q():
    atoms = read(AG_CLUSTERS, index=0)

    model = _make_model("film")

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


def test_charge_conditioning_latent_differs_with_Q():
    atoms = read(AG_CLUSTERS, index=0)

    model = _make_model("latent")

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


def test_lorem_default_charge_conditioning_is_film():
    assert Lorem().charge_conditioning == "film"


def test_bec_default_charge_conditioning_is_film():
    assert LoremBEC().charge_conditioning == "film"


def test_bec_charge_conditioning_film_differs_with_Q():
    model = LoremBEC(
        cutoff=5.0,
        num_features=8,
        num_spherical_features=2,
        num_radial=4,
        charge_conditioning="film",
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


def test_bec_charge_conditioning_none_ignores_Q():
    model = LoremBEC(
        cutoff=5.0,
        num_features=8,
        num_spherical_features=2,
        num_radial=4,
        charge_conditioning="none",
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

    np.testing.assert_allclose(
        calc_plus.results["energy"], calc_minus.results["energy"], atol=1e-6
    )


def test_ag_clusters_smoke_all_modes():
    """Sanity check on the real (cropped) Ag3+/Ag3- dataset: finite E/F for
    every charge_conditioning mode, using each structure's own total_charge
    exactly as it flows through atoms.info -> to_sample -> to_batch."""
    frames = read(AG_CLUSTERS, index=":")
    assert len(frames) > 0
    assert all("total_charge" in atoms.info for atoms in frames)

    for mode in ["none", "film", "latent"]:
        model = _make_model(mode)
        calc = Calculator.from_model(model)
        for atoms in frames[:3]:
            calc.calculate(atoms)
            assert np.all(np.isfinite(calc.results["energy"]))
            assert np.all(np.isfinite(calc.results["forces"]))
            assert calc.results["forces"].shape == (len(atoms), 3)
