import numpy as np
import jax
import jax.numpy as jnp

import e3x
from ase.build import bulk, molecule
from ase.calculators.singlepoint import SinglePointCalculator

from lorem.batching import to_batch, to_sample
from lorem.calculator import Calculator
from lorem.models.backbone import ChargeEmbedding, field_magnitude, spherical_field
from lorem.models.bec import LoremBEC
from lorem.models.mlip import Lorem

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


def test_total_charge_survives_marathon_prepare_roundtrip(tmp_path):
    # marathon.grain.prepare() only persists atoms.info entries that are
    # declared in its own `properties` dict (storage="atoms.info") — unlike
    # to_sample()/to_batch() above, which always read atoms.info directly.
    # Real training datasets go through this prepare()/DataSource path, so
    # total_charge must be declared here or it's silently dropped.
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
    import lorem.batching as batching

    batching._warned_missing_total_charge = False
    try:
        for _ in range(3):
            to_sample(molecule("H2O"), cutoff=5.0, energy=False, forces=False, stress=False)
    finally:
        batching._warned_missing_total_charge = False

    out = capsys.readouterr().out
    assert out.count("not set; assuming") == 1


# -- data plumbing: atoms.info["external_field"] -> batch.external_field --


def test_external_field_flows_through_batch():
    atoms = molecule("H2O")
    atoms.info["external_field"] = [0.1, -0.2, 0.3]
    sample = to_sample(atoms, cutoff=5.0, energy=False, forces=False, stress=False)
    batch = to_batch([sample], [])
    assert np.allclose(batch.external_field[0], [0.1, -0.2, 0.3])


def test_external_field_defaults_to_zero():
    atoms = molecule("H2O")
    sample = to_sample(atoms, cutoff=5.0, energy=False, forces=False, stress=False)
    batch = to_batch([sample], [])
    assert np.allclose(batch.external_field[0], [0.0, 0.0, 0.0])


def test_external_field_survives_marathon_prepare_roundtrip(tmp_path):
    # same rationale as total_charge's equivalent test: marathon.grain.prepare()
    # only persists atoms.info entries declared in its own `properties` dict.
    from marathon.grain import DataSource, prepare

    def make(field):
        atoms = molecule("H2O")
        atoms.info["external_field"] = field
        atoms.calc = SinglePointCalculator(
            atoms, energy=0.0, forces=np.zeros((len(atoms), 3))
        )
        return atoms

    properties = {
        "energy": {"shape": (1,), "storage": "atoms.calc"},
        "forces": {"shape": ("atom", 3), "storage": "atoms.calc"},
        "external_field": {"shape": (3,), "storage": "atoms.info"},
    }

    prepare(
        [make([1.0, 0.0, 0.0]), make([0.0, -1.0, 0.0])],
        folder=tmp_path / "ds",
        properties=properties,
    )

    src = DataSource(tmp_path / "ds")
    values = sorted(
        tuple(np.asarray(src[i].info["external_field"]).tolist()) for i in range(len(src))
    )
    assert values == [(0.0, -1.0, 0.0), (1.0, 0.0, 0.0)]


def test_missing_external_field_warns_once(capsys):
    import lorem.batching as batching

    batching._warned_missing_external_field = False
    try:
        for _ in range(3):
            to_sample(molecule("H2O"), cutoff=5.0, energy=False, forces=False, stress=False)
    finally:
        batching._warned_missing_external_field = False

    out = capsys.readouterr().out
    assert out.count("not set; assuming") == 1


# -- ChargeEmbedding --


def test_charge_embedding_changes_with_Q():
    key = jax.random.key(0)
    num_atoms, d = 4, 6
    x = jax.random.normal(key, (num_atoms, d))
    atom_mask = jnp.ones(num_atoms, dtype=bool)
    Q_i = jnp.array([1.0, 1.0, -1.0, -1.0])

    model = ChargeEmbedding(features=d)
    params = model.init(key, Q_i, x, atom_mask)
    y = model.apply(params, Q_i, x, atom_mask)
    y_zero_Q = model.apply(params, jnp.zeros(num_atoms), x, atom_mask)

    assert y.shape == (num_atoms, d)
    assert not jnp.allclose(y, y_zero_Q)


# -- spherical_field / field_magnitude --


def test_spherical_field_l0_is_always_zero():
    E_i = jnp.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-5.0, 0.1, 0.0]])
    harmonics = spherical_field(E_i)
    assert np.allclose(harmonics[..., 0, :], 0.0)


def test_spherical_field_l1_block_is_cartesian_order():
    E_i = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    harmonics = spherical_field(E_i)
    # harmonics: [num_atoms, 1, 4, 1] -- lm indices 1,2,3 should be x,y,z
    l1 = harmonics[:, 0, 1:4, 0]
    assert np.allclose(l1, np.eye(3), atol=1e-5)


def test_field_magnitude():
    E_i = jnp.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])
    mag = field_magnitude(E_i)
    assert np.allclose(mag, [5.0, 0.0], atol=1e-4)
    assert np.all(np.isfinite(mag))


# -- end-to-end: Lorem/LoremBEC on hand-built water molecules --


def _make_model(field_conditioning="none", lr=False):
    return Lorem(
        cutoff=6.0,
        num_features=8,
        num_spherical_features=2,
        num_radial=4,
        num_message_passing=1,
        lr=lr,
        field_conditioning=field_conditioning,
    )


def test_charge_conditioning_differs_with_Q():
    atoms = molecule("H2O")

    model = _make_model()

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


def test_bec_charge_conditioning_differs_with_Q():
    model = LoremBEC(
        cutoff=5.0,
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
    """Sanity check across charge states: finite E/F for a hand-built water
    molecule at Q in {-1, 0, +1}, using total_charge exactly as it flows
    through atoms.info -> to_sample -> to_batch."""
    model = _make_model()
    calc = Calculator.from_model(model)
    for q in (-1.0, 0.0, 1.0):
        atoms = molecule("H2O")
        atoms.info["total_charge"] = q
        calc.calculate(atoms)
        assert np.all(np.isfinite(calc.results["energy"]))
        assert np.all(np.isfinite(calc.results["forces"]))
        assert calc.results["forces"].shape == (len(atoms), 3)


# -- mode-switch sanity: Lorem field conditioning --


def test_field_conditioning_none_ignores_field():
    model = _make_model(field_conditioning="none")

    atoms = molecule("H2O")
    atoms.info["total_charge"] = 0.0

    atoms_zero = atoms.copy()
    atoms_zero.info["external_field"] = [0.0, 0.0, 0.0]
    calc_zero = Calculator.from_model(model)
    calc_zero.calculate(atoms_zero)

    atoms_field = atoms.copy()
    atoms_field.info["external_field"] = [0.3, -0.2, 0.5]
    calc_field = Calculator.from_model(model)
    calc_field.calculate(atoms_field)

    assert np.allclose(calc_zero.results["energy"], calc_field.results["energy"], atol=1e-6)


def test_field_conditioning_l1_changes_with_field():
    model = _make_model(field_conditioning="l1")

    atoms = molecule("H2O")
    atoms.info["total_charge"] = 0.0

    atoms_zero = atoms.copy()
    atoms_zero.info["external_field"] = [0.0, 0.0, 0.0]
    calc_zero = Calculator.from_model(model)
    calc_zero.calculate(atoms_zero)

    atoms_field = atoms.copy()
    atoms_field.info["external_field"] = [0.3, -0.2, 0.5]
    calc_field = Calculator.from_model(model)
    calc_field.calculate(atoms_field)

    assert not np.allclose(
        calc_zero.results["energy"], calc_field.results["energy"], atol=1e-6
    )


def test_field_conditioning_l1_l0_changes_with_field():
    model = _make_model(field_conditioning="l1_l0")

    atoms = molecule("H2O")
    atoms.info["total_charge"] = 0.0

    atoms_zero = atoms.copy()
    atoms_zero.info["external_field"] = [0.0, 0.0, 0.0]
    calc_zero = Calculator.from_model(model)
    calc_zero.calculate(atoms_zero)

    atoms_field = atoms.copy()
    atoms_field.info["external_field"] = [0.3, -0.2, 0.5]
    calc_field = Calculator.from_model(model)
    calc_field.calculate(atoms_field)

    assert not np.allclose(
        calc_zero.results["energy"], calc_field.results["energy"], atol=1e-6
    )


def test_field_conditioning_finite_at_zero_field():
    for mode in ("none", "l1", "l1_l0"):
        model = _make_model(field_conditioning=mode)
        calc = Calculator.from_model(model)
        atoms = molecule("H2O")
        atoms.info["total_charge"] = 0.0
        atoms.info["external_field"] = [0.0, 0.0, 0.0]
        calc.calculate(atoms)
        assert np.all(np.isfinite(calc.results["energy"]))
        assert np.all(np.isfinite(calc.results["forces"]))


# -- rotation equivariance (the important one) --


def test_field_conditioning_l1_is_rotation_equivariant():
    R = np.array(e3x.so3.random_rotation(jax.random.key(0)))
    model = _make_model(field_conditioning="l1")

    atoms = molecule("H2O")
    atoms.info["total_charge"] = 0.0
    atoms.info["external_field"] = [0.3, -0.2, 0.5]
    calc = Calculator.from_model(model)
    calc.calculate(atoms)

    atoms_rot = atoms.copy()
    atoms_rot.positions = atoms.positions @ R.T
    atoms_rot.info["external_field"] = np.array(atoms.info["external_field"]) @ R.T
    calc_rot = Calculator.from_model(model)
    calc_rot.calculate(atoms_rot)

    assert np.allclose(calc.results["energy"], calc_rot.results["energy"], atol=1e-4)
    assert np.allclose(calc.results["forces"] @ R.T, calc_rot.results["forces"], atol=1e-4)


def test_field_conditioning_l1_l0_is_rotation_equivariant():
    R = np.array(e3x.so3.random_rotation(jax.random.key(0)))
    model = _make_model(field_conditioning="l1_l0")

    atoms = molecule("H2O")
    atoms.info["total_charge"] = 0.0
    atoms.info["external_field"] = [0.3, -0.2, 0.5]
    calc = Calculator.from_model(model)
    calc.calculate(atoms)

    atoms_rot = atoms.copy()
    atoms_rot.positions = atoms.positions @ R.T
    atoms_rot.info["external_field"] = np.array(atoms.info["external_field"]) @ R.T
    calc_rot = Calculator.from_model(model)
    calc_rot.calculate(atoms_rot)

    assert np.allclose(calc.results["energy"], calc_rot.results["energy"], atol=1e-4)
    assert np.allclose(calc.results["forces"] @ R.T, calc_rot.results["forces"], atol=1e-4)


# -- direction sensitivity without rotating the structure (catches an
# accidentally-invariant-only coupling, e.g. one that only picks up |E|) --


def test_field_conditioning_l1_direction_sensitive():
    model = _make_model(field_conditioning="l1")

    atoms = molecule("H2O")
    atoms.info["total_charge"] = 0.0

    atoms_a = atoms.copy()
    atoms_a.info["external_field"] = [0.4, 0.0, 0.0]
    calc_a = Calculator.from_model(model)
    calc_a.calculate(atoms_a)

    atoms_b = atoms.copy()
    atoms_b.info["external_field"] = [0.0, 0.4, 0.0]
    calc_b = Calculator.from_model(model)
    calc_b.calculate(atoms_b)

    assert not np.allclose(calc_a.results["energy"], calc_b.results["energy"], atol=1e-6)


def test_field_conditioning_l1_l0_direction_sensitive():
    model = _make_model(field_conditioning="l1_l0")

    atoms = molecule("H2O")
    atoms.info["total_charge"] = 0.0

    atoms_a = atoms.copy()
    atoms_a.info["external_field"] = [0.4, 0.0, 0.0]
    calc_a = Calculator.from_model(model)
    calc_a.calculate(atoms_a)

    atoms_b = atoms.copy()
    atoms_b.info["external_field"] = [0.0, 0.4, 0.0]
    calc_b = Calculator.from_model(model)
    calc_b.calculate(atoms_b)

    assert not np.allclose(calc_a.results["energy"], calc_b.results["energy"], atol=1e-6)


# -- field-reversal asymmetry (a norm-only readout is structurally even in
# the field, ||x|| == ||-x||, and can never represent the dominant linear
# dipole-field term E ~ -mu.F; these catch a regression back to that state) --


def test_field_conditioning_l1_not_symmetric_under_field_reversal():
    model = _make_model(field_conditioning="l1")

    atoms = molecule("H2O")
    atoms.info["total_charge"] = 0.0

    atoms_pos = atoms.copy()
    atoms_pos.info["external_field"] = [0.3, -0.2, 0.5]
    calc_pos = Calculator.from_model(model)
    calc_pos.calculate(atoms_pos)

    atoms_neg = atoms.copy()
    atoms_neg.info["external_field"] = [-0.3, 0.2, -0.5]
    calc_neg = Calculator.from_model(model)
    calc_neg.calculate(atoms_neg)

    assert not np.allclose(
        calc_pos.results["energy"], calc_neg.results["energy"], atol=1e-6
    )


def test_field_conditioning_l1_l0_not_symmetric_under_field_reversal():
    model = _make_model(field_conditioning="l1_l0")

    atoms = molecule("H2O")
    atoms.info["total_charge"] = 0.0

    atoms_pos = atoms.copy()
    atoms_pos.info["external_field"] = [0.3, -0.2, 0.5]
    calc_pos = Calculator.from_model(model)
    calc_pos.calculate(atoms_pos)

    atoms_neg = atoms.copy()
    atoms_neg.info["external_field"] = [-0.3, 0.2, -0.5]
    calc_neg = Calculator.from_model(model)
    calc_neg.calculate(atoms_neg)

    assert not np.allclose(
        calc_pos.results["energy"], calc_neg.results["energy"], atol=1e-6
    )


# -- lr on/off cross-check --


def test_field_conditioning_l1_with_lr_finite_and_direction_sensitive():
    model = _make_model(field_conditioning="l1", lr=True)

    atoms = molecule("H2O")
    atoms.info["total_charge"] = 0.0

    atoms_a = atoms.copy()
    atoms_a.info["external_field"] = [0.4, 0.0, 0.0]
    calc_a = Calculator.from_model(model)
    calc_a.calculate(atoms_a)
    assert np.all(np.isfinite(calc_a.results["energy"]))
    assert np.all(np.isfinite(calc_a.results["forces"]))

    atoms_b = atoms.copy()
    atoms_b.info["external_field"] = [0.0, 0.4, 0.0]
    calc_b = Calculator.from_model(model)
    calc_b.calculate(atoms_b)

    assert not np.allclose(calc_a.results["energy"], calc_b.results["energy"], atol=1e-6)
