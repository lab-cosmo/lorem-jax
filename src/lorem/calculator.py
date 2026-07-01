import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_default_matmul_precision", "float32")

from ase.calculators.calculator import (
    BaseCalculator,
    PropertyNotImplementedError,
)
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
        cutoff=None,
        atoms=None,
        stress=False,
        bec=False,
        add_offset=True,
        double_precision=False,
        skin=0.25,
        num_k=None,
        lr_wavelength=None,
        smearing=None,
    ):
        self.params = params
        self.cutoff = cutoff
        self.num_k = num_k
        self.lr_wavelength = lr_wavelength
        self.smearing = smearing
        self.skin = skin
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
        kwargs.setdefault("cutoff", model.cutoff)
        kwargs.setdefault("num_k", getattr(model, "num_k", None))
        return cls(model.predict, species_weights, params, **kwargs)

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
        kwargs.setdefault("cutoff", model.cutoff)
        kwargs.setdefault("num_k", getattr(model, "num_k", None))
        return cls(model.predict, species_to_weight, params, **kwargs)

    def update(self, atoms):
        if self._nl_cache.needs_update(atoms):
            # Structural change or combined displacement beyond skin
            self.results = {}
            self.atoms = atoms.copy()
            self.setup(atoms)
        elif self.atoms is None or not self._geometry_unchanged(atoms):
            # Positions and/or cell changed but within skin budget
            self.results = {}
            self.atoms = atoms.copy()
            self._update_geometry(atoms)

    def _geometry_unchanged(self, atoms):
        return np.array_equal(
            atoms.get_positions(), self.atoms.get_positions()
        ) and np.array_equal(atoms.get_cell()[:], self.atoms.get_cell()[:])

    def setup(self, atoms):
        from lorem.batching import to_batch, to_sample

        sample = to_sample(
            atoms,
            cutoff=self.cutoff,
            num_k=self.num_k,
            lr_wavelength=self.lr_wavelength,
            smearing=self.smearing,
            energy=False,
            forces=False,
            stress=False,
        )
        batch = to_batch([sample], [])
        self.batch = jax.tree.map(lambda x: jnp.array(x), batch)

        # Cover both neighbor lists: the Ewald list may use a larger (num_k
        # derived) cutoff than the short-range one, so its cell shifts can be
        # larger. The cache must search far enough to rebuild both correctly.
        max_cell_shift = max(
            int(np.abs(np.array(self.batch.mlip.cell_shifts)).max()),
            int(np.abs(np.array(self.batch.realspace.cell_shifts)).max()),
        )
        self._nl_cache.save_reference(atoms, max_cell_shift=max_cell_shift)

    def _update_geometry(self, atoms):
        """Update positions and cell in cached batch without rebuilding
        the neighbor lists. Both the short-range and the Ewald parts share
        realspace.positions/realspace.cell: the model recomputes R_ij from
        them, and the Ewald calculator recomputes k-vectors from realspace.cell
        (pbc.k_grid stores only integer frequency indices). So forces, energy,
        and stress remain correct."""
        realspace = self.batch.realspace
        n_atoms = len(atoms)

        positions = np.zeros(np.array(realspace.positions).shape, dtype=np.float32)
        positions[:n_atoms] = atoms.get_positions()

        cell = np.array(realspace.cell)
        new_cell = atoms.get_cell()[:].astype(np.float32)
        if atoms.get_pbc().sum() == 2:
            from jaxpme.batched_mixed.batching import shrink_2d_cell

            new_cell = shrink_2d_cell(new_cell, atoms.get_pbc(), positions[:n_atoms])
        cell[0] = new_cell

        new_realspace = realspace._replace(
            positions=jnp.array(positions),
            cell=jnp.array(cell),
        )
        self.batch = self.batch._replace(realspace=new_realspace)

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
                    results[key][self.batch.realspace.structure_mask].squeeze()
                )
            elif key == "forces":
                actual_results[key] = np.array(
                    results[key][self.batch.realspace.atom_mask].reshape(-1, 3),
                    dtype=np.float32,
                )
            elif key == "stress":
                virial = np.array(
                    results[key][self.batch.realspace.structure_mask].reshape(3, 3),
                    dtype=np.float32,
                )
                volume = atoms.get_volume()
                from ase.stress import full_3x3_to_voigt_6_stress

                actual_results[key] = full_3x3_to_voigt_6_stress(virial / volume)

        # BEC passthrough: when model outputs "apt" (e.g. LoremBEC), expose as
        # "born_effective_charges" in (natoms, 3, 3) layout for ase compatibility
        if "apt" in results:
            actual_results["born_effective_charges"] = np.array(
                results["apt"][self.batch.realspace.atom_mask].reshape(-1, 3, 3),
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


def _model_predicts_bec(model):
    from lorem.models.bec import LoremBEC

    return isinstance(model, LoremBEC)
