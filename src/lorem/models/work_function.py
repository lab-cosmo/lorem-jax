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

# Phi = dE/dq, with q the total charge in units of +e and E the total energy,
# not the grand potential. The textbook Phi = -E_F carries a minus because E_F
# = dE/dN_e counts electrons; q counts the holes, dN_e = -dq, and the two
# minuses cancel. So there is no sign flip anywhere below, and datasets must
# supply `work_function` labels in this convention.


class LoremWF(nn.Module):
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
    # False routes Phi to its own pooled head instead of the energy derivative
    work_function_from_energy: bool = True

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
        # `stages` mirrors the residual energy readouts below: one entry per
        # stage, so the direct head sees the same features the energy does.
        stages = [nodes_scalar]
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
            stages.append(nodes_scalar)
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
            stages.append(nodes_scalar)
            energy += masked(MLP(features=[d, d, 1]), nodes_scalar, atom_mask)[..., 0]

        if self.work_function_from_energy:
            work_function = None
        else:
            work_function = PooledScalarHead(features=d)(stages, sr)

        return energy, work_function

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
        energies, work_function = self.apply(
            params,
            batch.atomic_numbers,
            batch.sr,
            batch.nopbc,
            batch.pbc,
            batch.total_charge,
        )
        energies *= sr.atom_mask

        # only the sum is differentiated, so the head cannot reach the forces;
        # it still trains, through `aux`
        return jnp.sum(energies), (energies, work_function)

    def predict(self, params, batch, stress=False):
        sr = batch[1]

        energy_and_derivatives_fn = jax.value_and_grad(
            self.energy, allow_int=True, has_aux=True, argnums=1
        )
        batch_energy_and_aux, batch_grads = energy_and_derivatives_fn(params, batch)
        _, (energies, work_function) = batch_energy_and_aux

        grads = batch_grads.sr

        energy = (
            jax.ops.segment_sum(energies, sr.atom_to_structure, sr.cell.shape[0])
            * sr.structure_mask
        )

        forces = -grads.positions

        if self.work_function_from_energy:
            # structures in a batch do not interact, so d(sum)/dq_s is already
            # structure s's own dE/dq. The per-species baseline is
            # charge-independent, so this needs no offset unlike the energy.
            work_function = batch_grads.total_charge

        results = {
            "energy": energy,
            "forces": forces,
            "work_function": work_function * sr.structure_mask,
        }

        if stress:
            stress = (
                jax.ops.segment_sum(
                    jnp.einsum("ia,ib->iab", sr.positions, grads.positions),
                    sr.atom_to_structure,
                    num_segments=sr.cell.shape[0],
                )
                + jnp.einsum("sAa,sAb->sab", sr.cell, grads.cell)
            ) * sr.structure_mask[:, None, None]
            results["stress"] = stress

        return results


# CP-MACE's Fermi-level readout: every stage's invariant node features
# concatenated, pooled over atoms, one MLP (Wang et al., J. Chem. Theory
# Comput. 21, 7628 (2025), eq. 8)
class PooledScalarHead(nn.Module):
    features: int

    @nn.compact
    def __call__(self, stages, sr):
        num_structures = sr.cell.shape[0]
        atom_to_structure = sr.atom_to_structure

        x = jnp.concatenate(stages, axis=-1) * sr.atom_mask[..., None]

        counts = jax.ops.segment_sum(
            sr.atom_mask.astype(x.dtype), atom_to_structure, num_structures
        )
        # mean, not the sum the energy readout uses: Phi is intensive
        pooled = (
            jax.ops.segment_sum(x, atom_to_structure, num_structures)
            / jnp.maximum(counts, 1.0)[..., None]
        )

        pooled = masked(nn.LayerNorm(), pooled, sr.structure_mask)
        phi = masked(
            MLP(features=[self.features, self.features, 1]),
            pooled,
            sr.structure_mask,
        )[..., 0]

        return phi * sr.structure_mask
