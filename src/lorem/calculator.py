import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_default_matmul_precision", "float32")

from ase.calculators.abc import GetPropertiesMixin
from ase.calculators.calculator import PropertyNotImplementedError, compare_atoms


class Calculator(GetPropertiesMixin):
    name = "lorem"
    parameters = {}

    def todict(self):
        return self.parameters

    implemented_properties = [
        "energy",
        "forces",
        "stress",
    ]

    def __init__(
        self,
        pred_fn,
        species_weights,
        params,
        cutoff,
        atoms=None,
        stress=False,
        add_offset=True,
        double_precision=False,
    ):
        self.params = params
        self.cutoff = cutoff
        self.add_offset = add_offset
        self.double_precision = double_precision

        if not stress:
            self.implemented_properties = ["energy", "forces"]

        predict_fn = lambda params, batch: pred_fn(params, batch, stress=stress)

        self.predict_fn = jax.jit(predict_fn)
        self.species_weights = species_weights

        self.atoms = None
        self.batch = None
        self.results = {}
        if atoms is not None:
            self.setup(atoms)

    @classmethod
    def from_model(cls, model, params=None, species_weights=None, **kwargs):
        """Create a Calculator from a Lorem model instance."""
        if params is None:
            params = model.init(jax.random.key(0), *model.dummy_inputs())
        if species_weights is None:
            species_weights = {}
            kwargs.setdefault("add_offset", False)
        return cls(
            model.predict, species_weights, params, model.cutoff, **kwargs
        )

    @classmethod
    def from_checkpoint(
        cls,
        folder,
        **kwargs,
    ):
        from pathlib import Path

        from marathon.io import from_dict, read_yaml

        folder = Path(folder)

        model = from_dict(read_yaml(folder / "model/model.yaml"))

        _ = model.init(jax.random.key(1), *model.dummy_inputs())

        baseline = read_yaml(folder / "model/baseline.yaml")
        species_to_weight = baseline["elemental"]

        from marathon.emit.checkpoint import read_msgpack

        params = read_msgpack(folder / "model/model.msgpack")

        return cls(
            model.predict, species_to_weight, params, model.cutoff, **kwargs
        )

    def update(self, atoms):
        changes = compare_atoms(self.atoms, atoms)

        if len(changes) > 0:
            self.results = {}
            self.atoms = atoms.copy()
            self.setup(atoms)

    def setup(self, atoms):
        from lorem.batching import to_batch, to_sample

        sample = to_sample(
            atoms, self.cutoff, energy=False, forces=False, stress=False
        )
        batch = to_batch([sample], [])
        self.batch = jax.tree.map(lambda x: jnp.array(x), batch)

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=None,
        **kwargs,
    ):
        self.update(atoms)

        results = self.predict_fn(self.params, self.batch)

        actual_results = {}
        for key in self.implemented_properties:
            if key == "energy":
                actual_results[key] = float(
                    results[key][self.batch.sr.structure_mask].squeeze()
                )
            elif key == "forces":
                actual_results[key] = np.array(
                    results[key][self.batch.sr.atom_mask].reshape(-1, 3),
                    dtype=np.float32,
                )
            elif key == "stress":
                raise KeyError

        if self.add_offset:
            energy_offset = np.sum(
                [
                    self.species_weights[Z]
                    for Z in atoms.get_atomic_numbers()
                ]
            )
            actual_results["energy"] += energy_offset

        self.results = actual_results
        return actual_results

    def get_property(self, name, atoms=None, allow_calculation=True):
        if name not in self.implemented_properties:
            raise PropertyNotImplementedError(
                f"{name} property not implemented"
            )

        self.update(atoms)

        if name not in self.results:
            if not allow_calculation:
                return None
            self.calculate(atoms=atoms)

        if name not in self.results:
            # For some reason the calculator was not able to do what we want,
            # and that is OK.
            raise PropertyNotImplementedError(
                f"{name} property not present in results!"
            )

        result = self.results[name]
        if isinstance(result, np.ndarray):
            result = result.copy()
        return result

    def get_potential_energy(self, atoms=None, force_consistent=True):
        # force_consistent is ignored; we are always consistent
        return self.get_property(name="energy", atoms=atoms)
