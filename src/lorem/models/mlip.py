import jax
import jax.numpy as jnp

import e3x
import flax.linen as nn
from jaxpme.batched_mixed import Ewald

from lorem.models.backbone import (
    MLP,
    ChargeConditioning,
    Initial,
    RadialCoefficients,
    Update,
    degree_wise_repeat_last_axis,
    spherical_norm_last_axis,
)
from lorem.models.backbone import (
    _masked as masked,
)
from lorem.transforms import ToBatch, ToSample

# vacuum permittivity in e^2 / (eV * Angstrom)
EPSILON_0 = 0.005526349406


class Lorem(nn.Module):
    cutoff: float = 5.0
    max_degree: int = 6
    max_degree_lr: int = 2
    num_features: int = 128
    num_radial: int = 32
    num_species: int = 8
    num_spherical_features: int = 8
    cutoff_fn: str = "cosine_cutoff"
    radial_basis: str = "basic_bernstein"
    lr: bool = True
    num_message_passing: int = 0
    equivariant_message_passing: bool = True
    initialize_node_features: bool = True

    @property
    def to_batch(self):
        return ToBatch

    @property
    def to_sample(self):
        return ToSample

    @nn.compact
    def __call__(
        self,
        Z_i,
        sr,
        nopbc,
        pbc,
        Q,
    ):
        R = sr.positions
        i = sr.centers
        j = sr.others
        cell = sr.cell
        cell_shifts = sr.cell_shifts
        pair_mask = sr.pair_mask
        atom_mask = sr.atom_mask
        atom_to_structure = sr.atom_to_structure

        R_ij = (
            R[j] - R[i] + jnp.einsum("pA,pAa->pa", cell_shifts, cell[sr.pair_to_structure])
        )

        num_atoms = Z_i.shape[0]
        num_pairs = R_ij.shape[0]

        max_degree = self.max_degree
        max_degree_lr = self.max_degree_lr
        num_l = self.max_degree + 1
        num_lm = int((self.max_degree + 1) ** 2)

        d = self.num_features
        s = self.num_spherical_features

        Q_i = Q[atom_to_structure] * atom_mask

        # empirical factors to make var of equivariant norm more uniform across l
        l_factors = (
            jnp.array([(2 * l + 1) for l in range(max_degree + 1)], dtype=float) ** 0.25
        )

        # -- initial embeddings --
        radial, spherical, species, cutoffs, r_ij = Initial(
            cutoff=self.cutoff,
            max_degree=self.max_degree,
            num_features=self.num_features,
            num_radial=self.num_radial,
            num_species=self.num_species,
            num_spherical_features=self.num_spherical_features,
            cutoff_fn=self.cutoff_fn,
            radial_basis=self.radial_basis,
        )(
            R_ij,
            Z_i,
            pair_mask,
            atom_mask,
        )

        # -- learned linear transformation of radial expansion --
        edges_scalar = RadialCoefficients(d)(
            jnp.concatenate([species[i], species[j]], axis=-1),
            radial,
            cutoffs,
            pair_mask,
        )

        # -- initial scalar and equivariant (spherical) node features
        if self.initialize_node_features:
            nodes_scalar = masked(nn.Dense(d, use_bias=True), species, atom_mask)
        else:
            nodes_scalar = jnp.zeros((num_atoms, d), dtype=species.dtype)

        updates = (
            jax.ops.segment_sum(
                masked(nn.Dense(d, use_bias=False), edges_scalar, pair_mask),
                i,
                num_segments=num_atoms,
            )
            * atom_mask[..., None]
        )

        nodes_scalar = Update(d)(nodes_scalar, updates, atom_mask)
        nodes_scalar = ChargeConditioning(d)(Q_i, nodes_scalar, atom_mask)

        coefficients = masked(
            nn.Dense(num_l * s, use_bias=False), edges_scalar, pair_mask
        ).reshape(num_pairs, num_l, s)
        coefficients = degree_wise_repeat_last_axis(coefficients, max_degree)
        edges_spherical = jnp.einsum("plf,pl->plf", coefficients, spherical)

        nodes_spherical = (
            jax.ops.segment_sum(
                edges_spherical.reshape(num_pairs, 1, num_lm, s),
                i,
                num_segments=num_atoms,
            )
            * atom_mask[..., None, None, None]
        )
        nodes_spherical = e3x.nn.TensorDense(use_bias=False, include_pseudotensors=False)(
            nodes_spherical
        )

        # -- mix equivariant information into scalar node features --
        norms = spherical_norm_last_axis(nodes_spherical, max_degree)
        updates = (norms * l_factors[None, None, :, None]).reshape(num_atoms, -1)

        nodes_scalar = Update(d)(nodes_scalar, updates, atom_mask)

        # -- initial prediction --
        energy = masked(MLP(features=[d, d, 1]), nodes_scalar, atom_mask)[..., 0]

        # -- message passing (if turned on) --
        for _ in range(self.num_message_passing):
            edges_scalar = RadialCoefficients(d)(
                jnp.concatenate([nodes_scalar[i], nodes_scalar[j]], axis=-1),
                radial,
                cutoffs,
                pair_mask,
            )
            updates = (
                jax.ops.segment_sum(
                    masked(
                        nn.Dense(d, use_bias=False),
                        edges_scalar,
                        pair_mask,
                    ),
                    i,
                    num_segments=num_atoms,
                )
                * atom_mask[..., None]
            )

            nodes_scalar = Update(d)(nodes_scalar, updates, atom_mask)

            if self.equivariant_message_passing:
                coefficients = masked(
                    nn.Dense(num_l * s, use_bias=False),
                    edges_scalar,
                    pair_mask,
                ).reshape(num_pairs, num_l, s)
                coefficients = degree_wise_repeat_last_axis(coefficients, max_degree)
                edges_spherical = jnp.einsum(
                    "plf,pl->plf", coefficients, spherical
                ).reshape(num_pairs, 1, num_lm, s)

                messages = (
                    e3x.nn.MessagePass(include_pseudotensors=False)(
                        nodes_spherical,
                        edges_spherical,
                        dst_idx=i,
                        src_idx=j,
                    )
                    * atom_mask[..., None, None, None]
                )
                nodes_spherical = e3x.nn.Tensor(include_pseudotensors=False)(
                    e3x.nn.Dense(use_bias=False, features=s)(nodes_spherical),
                    e3x.nn.Dense(use_bias=False, features=s)(messages),
                )

                norms = spherical_norm_last_axis(nodes_spherical, max_degree)
                updates = (norms * l_factors[None, None, :, None]).reshape(num_atoms, -1)
                nodes_scalar = Update(d)(nodes_scalar, updates, atom_mask)

            # -- residual prediction --
            energy += masked(MLP(features=[d, d, 1]), nodes_scalar, atom_mask)[..., 0]

        if self.lr:
            # -- compute LR potentials --
            scalar_charges = masked(MLP(features=[2 * d, 1]), nodes_scalar, atom_mask)

            spherical_charges = e3x.nn.TensorDense(
                features=1,
                use_bias=False,
                max_degree=max_degree_lr,
                include_pseudotensors=False,
            )(nodes_spherical).reshape(num_atoms, -1)
            charges = jnp.concatenate([scalar_charges, spherical_charges], axis=-1)

            calculator = Ewald()
            potentials = jax.vmap(
                lambda q: calculator.potentials(q, sr, nopbc, pbc),
                in_axes=-1,
                out_axes=-1,
            )(charges)

            scalar_potential = potentials[..., 0][..., None]
            spherical_potential = potentials[..., 1:].reshape(num_atoms, 1, -1, 1)

            # -- combine LR potentials back into local features --
            spherical_potential = e3x.nn.Dense(s, use_bias=False)(spherical_potential)
            spherical_updates = e3x.nn.Tensor(include_pseudotensors=False)(
                spherical_potential, nodes_spherical
            )

            norms = spherical_norm_last_axis(spherical_updates, max_degree)
            norms = (norms * l_factors[None, None, :, None]).reshape(num_atoms, -1)
            updates = jnp.concatenate([scalar_potential, norms], axis=-1)
            nodes_scalar = Update(d)(nodes_scalar, updates, atom_mask)

            # -- residual prediction --
            energy += masked(MLP(features=[d, d, 1]), nodes_scalar, atom_mask)[..., 0]

        return energy

    def atoms_to_batch(self, atoms):
        from lorem.batching import to_batch, to_sample

        sample = to_sample(atoms, self.cutoff, energy=False, forces=False, stress=False)
        batch = to_batch([sample], [])

        return jax.tree.map(lambda x: jnp.array(x), batch)

    def dummy_inputs(self):
        from ase.build import bulk

        atoms = bulk("Ar") * [2, 2, 2]
        atoms.info["total_charge"] = 0.0

        return self.atoms_to_batch(atoms)[:-1]

    def energy(self, params, batch):
        sr = batch[1]
        energies = self.apply(
            params,
            batch.atomic_numbers,
            batch.sr,
            batch.nopbc,
            batch.pbc,
            batch.total_charge,
        )
        energies *= sr.atom_mask

        return jnp.sum(energies), energies

    def _energy_and_grads(self, params, batch):
        """One value_and_grad of the energy w.r.t. the whole batch.

        Split out so `LoremQ` can read dE/dq off the same gradient pytree that
        already carries the forces, rather than paying for a second backward
        pass to get it.
        """
        sr = batch[1]

        energy_and_derivatives_fn = jax.value_and_grad(
            self.energy, allow_int=True, has_aux=True, argnums=1
        )
        batch_energy_and_atom_energies, batch_grads = energy_and_derivatives_fn(
            params, batch
        )
        _, energies = batch_energy_and_atom_energies

        energy = (
            jax.ops.segment_sum(energies, sr.atom_to_structure, sr.cell.shape[0])
            * sr.structure_mask
        )
        forces = -batch_grads.sr.positions

        return energy, forces, batch_grads

    def _stress(self, sr, grads):
        return (
            jax.ops.segment_sum(
                jnp.einsum("ia,ib->iab", sr.positions, grads.positions),
                sr.atom_to_structure,
                num_segments=sr.cell.shape[0],
            )
            + jnp.einsum("sAa,sAb->sab", sr.cell, grads.cell)
        ) * sr.structure_mask[:, None, None]

    def predict(self, params, batch, stress=False):
        """Energy, forces and optionally stress -- the plain MLIP contract.

        Charge derivatives live on `LoremQ`. They are meaningless here: this
        class applies `ChargeConditioning` unconditionally and `total_charge`
        defaults to 0, so a model trained where q never varied would otherwise
        report an unconstrained dE/dq alongside energy and forces, with the
        same apparent standing.
        """
        sr = batch[1]
        energy, forces, batch_grads = self._energy_and_grads(params, batch)

        results = {"energy": energy, "forces": forces}

        if stress:
            results["stress"] = self._stress(sr, batch_grads.sr)

        return results


class LoremQ(Lorem):
    """`Lorem` for systems where the total charge is a real, varied input.

    Identical architecture -- it inherits `__call__` untouched, so a `LoremQ`
    checkpoint is weight-compatible with a `Lorem` one. The only difference is
    what `predict` exposes: the derivatives with respect to `total_charge`,
    which are only meaningful when q actually varied during training.

    - `work_function` = dE/dq. Free: it falls out of the same backward pass as
      the forces.
    - `bec_z` = -(A eps0) d2E/(dr dq), the Born effective charge. Costs a
      forward-over-reverse pass, so it stays behind `predict_bec`.
    """

    # off by default: unlike the work function this is not free, so only runs
    # that actually supervise it should pay for it
    predict_bec: bool = False

    def _bec_z(self, params, batch):
        """Born effective charges, as the mixed second derivative.

        The charge sets a surface charge density q/A, hence a field q/(A eps0)
        along the slab normal, and an atom with Born effective charge Z* feels
        F = Z* q/(A eps0). So the dimensionless Z* the datasets carry is

            Z* = (A eps0) dF/dq = -(A eps0) d2E/(dr dq)

        Verified against razor's 3-point stencil: a finite-difference dF/dq
        regressed on the `bec_z` label gives slope 1/(A eps0) = 2.19839 for
        A = 82.3108 A^2, intercept ~1e-11, correlation 1.00000.

        Structures in a batch don't interact, so a tangent of ones on the
        per-structure charge vector hands each atom its own structure's
        derivative -- the same argument that makes the work function a single
        backward pass. This one costs a forward-over-reverse pass on top.
        """
        sr = batch[1]

        def dE_dpositions(total_charge):
            def energy_of_positions(positions):
                shifted = batch._replace(
                    sr=sr._replace(positions=positions),
                    total_charge=total_charge,
                )
                return self.energy(params, shifted)[0]

            return jax.grad(energy_of_positions)(sr.positions)

        _, d2E_drdq = jax.jvp(
            dE_dpositions,
            (batch.total_charge,),
            (jnp.ones_like(batch.total_charge),),
        )

        # in-plane cell area; the out-of-plane vector varies per structure and
        # must not enter (pbc = T T F)
        area = jnp.linalg.norm(
            jnp.cross(sr.cell[:, 0, :], sr.cell[:, 1, :]), axis=-1
        )
        scale = (area * EPSILON_0)[sr.atom_to_structure][:, None]

        return -d2E_drdq * scale * sr.atom_mask[:, None]

    def predict(self, params, batch, stress=False):
        sr = batch[1]
        energy, forces, batch_grads = self._energy_and_grads(params, batch)

        # dE/dq. `energy` sums over structures that do not interact, so the
        # derivative w.r.t. the per-structure total_charge vector is already
        # each structure's own value. The per-species baseline is
        # charge-independent and drops out, so no offset correction applies
        # here (unlike for the energy itself).
        results = {
            "energy": energy,
            "forces": forces,
            "work_function": batch_grads.total_charge * sr.structure_mask,
        }

        if self.predict_bec:
            results["bec_z"] = self._bec_z(params, batch)

        if stress:
            results["stress"] = self._stress(sr, batch_grads.sr)

        return results
