import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_default_matmul_precision", "float32")

from ase.calculators.calculator import (
    BaseCalculator,
    PropertyNotImplementedError,
)
from marathon.data.properties import DEFAULT_PROPERTIES
from marathon.emit.checkpoint import read_msgpack
from marathon.io import from_dict, read_yaml

from lorem.neighborlist import NeighborListCache


class Calculator(BaseCalculator):
    name = "lorem"
    parameters = {}

    def todict(self):
        return self.parameters

    # default; rebuilt per-instance in __init__ from what the model predicts
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        pred_fn,
        species_weights,
        params,
        cutoff,
        atoms=None,
        stress=False,
        bec=False,
        add_offset=True,
        double_precision=False,
        skin=0.25,
        inputs=(),
        properties=None,
        lr_kernel=None,
    ):
        from lorem.longrange import get_kernel

        self.params = params
        self.cutoff = cutoff
        self.skin = skin
        self.lr_kernel = get_kernel(lr_kernel)
        self.inputs = tuple(inputs)
        self.properties = DEFAULT_PROPERTIES if properties is None else properties
        self.add_offset = add_offset
        self.double_precision = double_precision

        self._nl_cache = NeighborListCache(skin=skin)

        # only advertise what the model actually produces
        self.implemented_properties = ["energy", "forces"]
        if stress:
            self.implemented_properties.append("stress")
        if bec:
            self.implemented_properties.append("born_effective_charges")

        predict_fn = lambda params, batch: pred_fn(params, batch, stress=stress)

        self.predict_fn = jax.jit(predict_fn)
        self.species_weights = species_weights

        self.atoms = None
        self.batch = None
        self._input_values = None
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
        kwargs.setdefault("bec", _model_predicts_bec(model))
        kwargs.setdefault("inputs", getattr(model, "inputs", ()))
        return cls(model.predict, species_weights, params, model.cutoff, **kwargs)

    @classmethod
    def from_checkpoint(
        cls,
        folder,
        **kwargs,
    ):
        from pathlib import Path

        folder = Path(folder)

        model = from_dict(read_yaml(folder / "model/model.yaml"))

        _ = model.init(jax.random.key(1), *model.dummy_inputs())

        baseline = read_yaml(folder / "model/baseline.yaml")
        species_to_weight = baseline["elemental"]

        params = read_msgpack(folder / "model/model.msgpack")

        kwargs.setdefault("bec", _model_predicts_bec(model))
        kwargs.setdefault("inputs", getattr(model, "inputs", ()))
        kwargs.setdefault("properties", _checkpoint_properties(folder))
        kwargs.setdefault("lr_kernel", _checkpoint_lr_kernel(folder))
        return cls(model.predict, species_to_weight, params, model.cutoff, **kwargs)

    def update(self, atoms):
        if self._nl_cache.needs_update(atoms) or self._inputs_changed(atoms):
            # Structural change, displacement beyond skin, or new model inputs
            self.results = {}
            self.atoms = atoms.copy()
            self.setup(atoms)
        elif self.atoms is None or not self._geometry_unchanged(atoms):
            # Positions and/or cell changed but within skin budget
            self.results = {}
            self.atoms = atoms.copy()
            self._update_geometry(atoms)

    def _inputs_changed(self, atoms):
        if not self.inputs:
            return False
        values = self._read_inputs(atoms)
        return self._input_values is None or any(
            not np.array_equal(values[k], self._input_values[k], equal_nan=True)
            for k in self.inputs
        )

    def _read_inputs(self, atoms):
        from marathon.data.sample import read_properties

        return read_properties(
            atoms, self.inputs, float_dtype=np.float32, properties=self.properties
        )

    def _geometry_unchanged(self, atoms):
        return np.array_equal(
            atoms.get_positions(), self.atoms.get_positions()
        ) and np.array_equal(atoms.get_cell()[:], self.atoms.get_cell()[:])

    def setup(self, atoms):
        from lorem.batching import to_batch, to_sample
        from lorem.longrange import max_cell_shift

        # Both neighbor lists (SR and Ewald real space) are built with the
        # skin; the cutoffs that enter the model and the Ewald split are not.
        sample = to_sample(
            atoms,
            self.cutoff,
            keys=(),
            inputs=self.inputs,
            lr_kernel=self.lr_kernel,
            skin=self.skin,
            properties=self.properties,
        )
        batch = to_batch(
            [sample],
            [],
            inputs=self.inputs,
            properties=self.properties,
            lr_kernel=self.lr_kernel,
        )
        self.batch = jax.tree.map(lambda x: jnp.array(x), batch)
        self._input_values = self._read_inputs(atoms)

        # the Ewald list can reach further than the SR list (e.g. with num_k),
        # so the skin budget has to hold for the larger cell shifts of the two
        shift = max(
            int(np.abs(np.array(self.batch.sr.cell_shifts)).max()),
            max_cell_shift(self.batch.lr),
        )
        self._nl_cache.save_reference(atoms, max_cell_shift=shift)

    def _update_geometry(self, atoms):
        """Update positions and cell in cached batch without rebuilding
        the neighbor lists. The model recomputes R_ij from the updated
        geometry.positions and geometry.cell, and the Ewald calculator
        recomputes k-vectors from the cell (pbc.k_grid stores only integer
        frequency indices). So forces, energy, and stress remain correct."""
        from lorem.longrange import update_effective_cell

        geometry = self.batch.geometry
        n_atoms = len(atoms)

        positions = np.zeros(np.array(geometry.positions).shape, dtype=np.float32)
        positions[:n_atoms] = atoms.get_positions()

        cell = np.array(geometry.cell)
        new_cell = atoms.get_cell()[:].astype(np.float32)
        cell[0] = new_cell

        new_geometry = geometry._replace(
            positions=jnp.array(positions),
            cell=jnp.array(cell),
        )
        # for 2D pbc, the k-space cell follows the slab
        lr = update_effective_cell(
            self.batch.lr, new_cell, atoms.get_pbc(), positions[:n_atoms]
        )
        lr = lr._replace(effective_cell=jnp.array(lr.effective_cell))
        self.batch = self.batch._replace(geometry=new_geometry, lr=lr)

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
                    results[key][self.batch.geometry.structure_mask].squeeze()
                )
            elif key == "forces":
                actual_results[key] = np.array(
                    results[key][self.batch.geometry.atom_mask].reshape(-1, 3),
                    dtype=np.float32,
                )
            elif key == "stress":
                virial = np.array(
                    results[key][self.batch.geometry.structure_mask].reshape(3, 3),
                    dtype=np.float32,
                )
                volume = atoms.get_volume()
                from ase.stress import full_3x3_to_voigt_6_stress

                actual_results[key] = full_3x3_to_voigt_6_stress(virial / volume)

        # BEC passthrough: when model outputs "apt" (e.g. LoremBEC), expose as
        # "born_effective_charges" in (natoms, 3, 3) layout for ase compatibility
        if "apt" in results:
            actual_results["born_effective_charges"] = np.array(
                results["apt"][self.batch.geometry.atom_mask].reshape(-1, 3, 3),
                dtype=np.float32,
            )

        if self.add_offset:
            energy_offset = np.sum(
                [self.species_weights[Z] for Z in atoms.get_atomic_numbers()]
            )
            actual_results["energy"] += energy_offset

        self.results = actual_results
        return actual_results

    def get_property(self, name, atoms=None, allow_calculation=True):
        if name not in self.implemented_properties:
            raise PropertyNotImplementedError(f"{name} property not implemented")

        self.update(atoms)

        if name not in self.results:
            if not allow_calculation:
                return None
            self.calculate(atoms=atoms)

        if name not in self.results:
            # For some reason the calculator was not able to do what we want,
            # and that is OK.
            raise PropertyNotImplementedError(f"{name} property not present in results!")

        result = self.results[name]
        if isinstance(result, np.ndarray):
            result = result.copy()
        return result

    def get_potential_energy(self, atoms=None, force_consistent=True):
        # force_consistent is ignored; we are always consistent
        return self.get_property(name="energy", atoms=atoms)


def _checkpoint_properties(folder):
    # the training run records the dataset's properties in the checkpoint config
    config = folder / "config.yaml"
    if not config.is_file():
        return None
    properties = read_yaml(config).get("training_pipeline", {}).get("batcher", {})
    properties = properties.get("properties")
    if properties is None:
        return None
    return {k: {**v, "shape": tuple(v["shape"])} for k, v in properties.items()}


def _checkpoint_lr_kernel(folder):
    # the kernel the model was trained with; None (the default kernel) for runs
    # that did not set one, or that predate the setting
    config = folder / "config.yaml"
    if not config.is_file():
        return None
    spec = read_yaml(config).get("training_pipeline", {}).get("lr_kernel")
    if spec is None:
        return None
    return from_dict(spec, default_namespace="lorem.longrange")


def _model_predicts_bec(model):
    from lorem.models.bec import LoremBEC

    return isinstance(model, LoremBEC)
