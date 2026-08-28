import numpy as np
import jax
import jax.numpy as jnp

from ase.build import bulk, molecule
from ase.calculators.singlepoint import SinglePointCalculator

from lorem.batching import to_batch, to_sample
from lorem.calculator import Calculator
from lorem.models.backbone import ChargeConditioning
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


# -- ChargeConditioning --


def test_charge_embedding_changes_with_Q():
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


# -- end-to-end: Lorem/LoremBEC on hand-built water molecules --


def _make_model(lr=False):
    return Lorem(
        cutoff=6.0,
        num_features=8,
        num_spherical_features=2,
        num_radial=4,
        num_message_passing=1,
        lr=lr,
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


# -- Calculator cache invalidation: total_charge lives in atoms.info, not
# positions/cell, so a reused Calculator instance must detect changes to it
# independently of the neighbor-list/geometry cache (e.g. a charge sweep at
# fixed geometry) --


def test_calculator_picks_up_total_charge_change_at_fixed_geometry():
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
