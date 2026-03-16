import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_default_matmul_precision", "float32")

from ase.calculators.calculator import (
    BaseCalculator,
    PropertyNotImplementedError,
)

from lorem.neighborlist import NeighborListCache


class Calculator(BaseCalculator):
    name = "lorem"
    parameters = {}

    def todict(self):
        return self.parameters

    implemented_properties = [
        "born_effective_charges",
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
        skin=0.25,
    ):
        self.params = params
        self.cutoff = cutoff
        self.skin = skin
        self.add_offset = add_offset
        self.double_precision = double_precision

        self._nl_cache = NeighborListCache(skin=skin)

        if not stress:
            self.implemented_properties = ["born_effective_charges", "energy", "forces"]

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
        return cls(model.predict, species_weights, params, model.cutoff, **kwargs)

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

        return cls(model.predict, species_to_weight, params, model.cutoff, **kwargs)

    def update(self, atoms):
        if self._nl_cache.needs_update(atoms):
            # Structural change or displacement beyond skin — full rebuild
            self.results = {}
            self.atoms = atoms.copy()
            self.setup(atoms)
        elif self.atoms is None or not np.array_equal(
            atoms.get_positions(), self.atoms.get_positions()
        ):
            # Positions changed but within skin — update positions only
            self.results = {}
            self.atoms = atoms.copy()
            self._update_positions(atoms)

    def setup(self, atoms):
        from lorem.batching import to_batch, to_sample

        nl_cutoff = self.cutoff + self.skin

        # Derive Ewald parameters from physical cutoff so the long-range
        # decomposition is unchanged when using the extended cutoff.
        lr_wavelength = self.cutoff / 8.0
        smearing = lr_wavelength * 2.0

        sample = to_sample(
            atoms,
            nl_cutoff,
            lr_wavelength=lr_wavelength,
            smearing=smearing,
            energy=False,
            forces=False,
            stress=False,
        )
        batch = to_batch([sample], [])
        self.batch = jax.tree.map(lambda x: jnp.array(x), batch)
        self._nl_cache.save_reference(atoms)

    def _update_positions(self, atoms):
        """Update positions in cached batch without rebuilding neighbor list."""
        sr = self.batch.sr
        n_atoms = len(atoms)
        positions = np.zeros(np.array(sr.positions).shape, dtype=np.float32)
        positions[:n_atoms] = atoms.get_positions()
        new_sr = sr._replace(positions=jnp.array(positions))
        self.batch = self.batch._replace(sr=new_sr)

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
                virial = np.array(
                    results[key][self.batch.sr.structure_mask].reshape(3, 3),
                    dtype=np.float32,
                )
                volume = atoms.get_volume()
                from ase.stress import full_3x3_to_voigt_6_stress

                actual_results[key] = full_3x3_to_voigt_6_stress(virial / volume)

        # BEC passthrough: when model outputs "apt" (e.g. LoremBEC), expose as
        # "born_effective_charges" in (natoms, 3, 3) layout for ase compatibility
        if "apt" in results:
            actual_results["born_effective_charges"] = np.array(
                results["apt"][self.batch.sr.atom_mask].reshape(-1, 3, 3),
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
