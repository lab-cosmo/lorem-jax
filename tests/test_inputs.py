import numpy as np
import jax.numpy as jnp

import pytest
from ase.build import bulk, molecule
from ase.calculators.singlepoint import SinglePointCalculator
from grain.python import Record, RecordMetadata
from marathon.data.properties import DEFAULT_PROPERTIES
from marathon.io import write_yaml

from lorem.batching import to_batch, to_sample
from lorem.calculator import Calculator
from lorem.models.bec import LoremBEC
from lorem.models.mlip import Lorem
from lorem.transforms import ToBatch, ToSample

PROPERTIES = {
    **DEFAULT_PROPERTIES,
    "total_charge": {"shape": (1,), "storage": "atoms.info"},
    "spins": {"shape": ("atom",), "storage": "atoms.arrays"},
}
INPUTS = ("total_charge", "spins")


def make_atoms(total_charge=None, spins=None):
    atoms = molecule("H2O")
    atoms.calc = SinglePointCalculator(atoms, energy=1.0, forces=np.zeros((3, 3)))
    if total_charge is not None:
        atoms.info["total_charge"] = total_charge
    if spins is not None:
        atoms.arrays["spins"] = np.asarray(spins, dtype=float)
    return atoms


def test_inputs_through_sample_and_batch():
    a = make_atoms(total_charge=1.0, spins=[1.0, -1.0, 0.5])
    b = make_atoms(total_charge=float("nan"), spins=[0.0, 0.0, 0.0])

    samples = [
        to_sample(x, cutoff=5.0, keys=["energy"], inputs=INPUTS, properties=PROPERTIES)
        for x in (a, b)
    ]
    assert "total_charge" in samples[0].structure
    assert "total_charge" not in samples[0].labels

    batch = to_batch(samples, ["energy"], inputs=INPUTS, properties=PROPERTIES)

    num_structures = batch.sr.cell.shape[0]
    num_atoms = batch.sr.positions.shape[0]
    assert batch.inputs["total_charge"].shape == (num_structures,)
    assert batch.inputs["spins"].shape == (num_atoms,)

    # per-structure: NaN -> zero with mask False, padding masked
    np.testing.assert_equal(batch.inputs["total_charge"][:2], [1.0, 0.0])
    np.testing.assert_equal(batch.inputs["total_charge_mask"][:2], [True, False])
    assert not batch.inputs["total_charge_mask"][2:].any()

    # per-atom: same layout as sr.positions, mask matches atom_mask
    np.testing.assert_equal(batch.inputs["spins"][:6], [1.0, -1.0, 0.5, 0, 0, 0])
    np.testing.assert_array_equal(batch.inputs["spins_mask"], batch.sr.atom_mask)

    assert "total_charge" not in batch.labels
    assert "energy" in batch.labels


def test_no_inputs_by_default():
    sample = to_sample(make_atoms(), cutoff=5.0)
    batch = to_batch([sample], ["energy"])
    assert batch.inputs == {}


def test_inputs_through_grain_transforms():
    atoms_list = [make_atoms(total_charge=q, spins=[q, 0, 0]) for q in (-1.0, 1.0)]

    to_sample_t = ToSample(
        cutoff=5.0, keys=("energy",), inputs=INPUTS, properties=PROPERTIES
    )
    records = [
        Record(RecordMetadata(index=i, record_key=i), to_sample_t.map(atoms))
        for i, atoms in enumerate(atoms_list)
    ]
    batcher = ToBatch(
        batch_size=4,
        keys=("energy",),
        inputs=INPUTS,
        properties=PROPERTIES,
        drop_remainder=False,
    )
    batch = next(iter(batcher(iter(records)))).data

    np.testing.assert_equal(batch.inputs["total_charge"][:2], [-1.0, 1.0])
    np.testing.assert_equal(batch.inputs["total_charge_mask"][:2], [True, True])
    np.testing.assert_equal(batch.inputs["spins"][:6], [-1.0, 0, 0, 1.0, 0, 0])


@pytest.mark.parametrize("cls", [Lorem, LoremBEC])
def test_model_declares_inputs(cls):
    model = cls(cutoff=5.0, num_features=8, num_spherical_features=2, num_radial=4)
    assert model.inputs == []
    assert len(model.dummy_inputs()) == 4

    model = cls(
        cutoff=5.0,
        num_features=8,
        num_spherical_features=2,
        num_radial=4,
        inputs=["total_charge"],
    )
    # init needs no inputs: atoms_to_batch is geometry only, inputs are wired up
    # by train.py and the Calculator, which know the dataset's properties
    assert model.atoms_to_batch(bulk("Ar") * [2, 2, 2]).inputs == {}
    assert len(model.dummy_inputs()) == 4


def test_calculator_reads_inputs_and_rebuilds_on_change():
    model = Lorem(
        cutoff=5.0,
        num_features=8,
        num_spherical_features=2,
        num_radial=4,
        inputs=["total_charge"],
    )
    # the model does not consume inputs yet, but the batch must carry them
    calc = Calculator.from_model(model, properties=PROPERTIES)
    assert calc.inputs == ("total_charge",)

    atoms = bulk("Ar") * [2, 2, 2]
    atoms.info["total_charge"] = 1.0
    calc.calculate(atoms)
    assert float(calc.batch.inputs["total_charge"][0]) == 1.0
    assert isinstance(calc.batch.inputs["total_charge"], jnp.ndarray)
    batch = calc.batch

    # same geometry, same charge: no rebuild
    calc.calculate(atoms)
    assert calc.batch is batch

    # same geometry, new charge: batch rebuilt with the new value
    atoms.info["total_charge"] = -1.0
    calc.calculate(atoms)
    assert float(calc.batch.inputs["total_charge"][0]) == -1.0
    assert calc.results["energy"] is not None


def test_calculator_without_inputs_ignores_atoms_info():
    model = Lorem(cutoff=5.0, num_features=8, num_spherical_features=2, num_radial=4)
    calc = Calculator.from_model(model)
    atoms = bulk("Ar") * [2, 2, 2]
    atoms.info["total_charge"] = 1.0
    calc.calculate(atoms)
    assert calc.batch.inputs == {}


def test_checkpoint_properties(tmp_path):
    from lorem.calculator import _checkpoint_properties

    assert _checkpoint_properties(tmp_path) is None

    write_yaml(
        tmp_path / "config.yaml",
        {
            "training_pipeline": {
                "batcher": {
                    "inputs": ["total_charge"],
                    "properties": {
                        "energy": {"shape": [1], "storage": "atoms.calc"},
                        "total_charge": {"shape": [1], "storage": "atoms.info"},
                    },
                }
            }
        },
    )
    properties = _checkpoint_properties(tmp_path)
    assert properties["total_charge"] == {"shape": (1,), "storage": "atoms.info"}
    assert properties["energy"]["shape"] == (1,)
