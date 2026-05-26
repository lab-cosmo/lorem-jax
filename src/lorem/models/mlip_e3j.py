"""LOREM with the e3j equivariant backend.

This is a *bit-exact* port of `lorem.models.mlip.Lorem`: each Flax parameter
has a one-to-one twin in the original e3x model. A trained e3x checkpoint
can be converted with `convert_e3x_params_to_e3j` and produce identical
energies / forces / stress.

How exact equivalence is achieved
---------------------------------
e3x and e3j use different conventions for spherical harmonics and Clebsch-
Gordan coefficients. We absorb the difference into deterministic per-path
constants:

- **SH**: use `e3x.so3.spherical_harmonics(cartesian_order=False)` directly.
  Cheap, and keeps e3x's normalisation throughout the model.
- **TP**: use `e3j.core.TensorProduct(sort=False)` for the CG sum. e3j
  uses e3nn-style CG, which differs from e3x's CG by a path-dependent
  factor `sqrt(2 * l_2 + 1)` (empirically verified). We absorb this into
  the collapse weights at the einsum step (and equivalently into the
  converter).
- **Collapse**: e3j keeps every CG path as a separate output multiplicity.
  We collapse with a hand-written einsum that multiplies each path by a
  feature-diagonal scalar and sums (matching `e3x.nn.Tensor`'s `(1,1,F)`
  per-path kernel slot exactly).

Kernel layouts in this module match e3x byte-for-byte (allowing for the
`sqrt(2 * l_2 + 1)` factor folded into the converter):

- `Dense` weights: `(features_in, features_out)` per l block — identical
  layout to `e3j.linen.Linear` with `channels=(F_in, F_out)`, just a transpose.
- `Tensor` weights: `(parity1, l1+1, parity2, l2+1, parity3, l3+1, F)` —
  identical to e3x's `Tensor.kernel`.

Only `include_pseudotensors=False` is supported (matches Lorem).
"""

import numpy as np
import jax
import jax.numpy as jnp

import functools

import e3j
import e3x
import flax.linen as nn
from e3x.ops import normalize_and_return_norm
from jaxpme.batched_mixed import Ewald

from lorem.models.backbone import (
    MLP,
    ChemicalEmbedding,
    RadialCoefficients,
    RadialEmbedding,
    Update,
)
from lorem.models.backbone import _masked as masked
from lorem.transforms import ToBatch, ToSample

# -- conventions -------------------------------------------------------------
#
# Parity (single, no pseudotensors): true tensors only, parity = (-1)**l.
# Arrays of equivariant features in this module are stored as
# `[N, F, irrep_dim]` where `irrep_dim = sum_l (2l+1)` and the m-axis is in
# e3j order (-l, -l+1, ..., +l).


def _scalar_irreps(max_degree: int) -> str:
    return " + ".join(f"1x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(max_degree + 1))


def _allowed_path(l1: int, l2: int, l3: int) -> bool:
    """True for parity-conserving paths under parity=(-1)**l convention."""
    if not (abs(l1 - l2) <= l3 <= l1 + l2):
        return False
    return (l1 + l2 + l3) % 2 == 0


def _enumerate_paths(L1: int, L2: int, L3: int):
    """Replicate e3j's `sort=False` iteration order over (l1, l2 -> l_out).

    Returns list of `(l1, l2, l_out)`. Restricted to `l_out <= L3` and to
    parity-conserving paths (`include_pseudotensors=False`).
    """
    paths = []
    for l1 in range(L1 + 1):
        for l2 in range(L2 + 1):
            for l_out in range(abs(l1 - l2), min(L3, l1 + l2) + 1):
                if _allowed_path(l1, l2, l_out):
                    paths.append((l1, l2, l_out))
    return paths


def _paths_by_l_out(L1: int, L2: int, L3: int):
    by_l = {l: [] for l in range(L3 + 1)}
    for l1, l2, l3 in _enumerate_paths(L1, L2, L3):
        by_l[l3].append((l1, l2))
    return by_l


@functools.cache
def _tensor_product(L1: int, L2: int, L3: int):
    """Channel-wise e3j TensorProduct in LEADING_CHANNELS layout, unsorted.

    Source irreps: `1x0e+...+1x{L1}` and `1x0e+...+1x{L2}`. Output filter
    selects irreps up to l_out = L3.
    """
    return e3j.core.TensorProduct(
        source=(_scalar_irreps(L1), _scalar_irreps(L2)),
        target=_scalar_irreps(L3),
        layout="LEADING_CHANNELS",
        mode="OUTER",
        sort=False,
    )


@functools.cache
def _path_slices(L1: int, L2: int, L3: int):
    """For each l_out (0..L3), the indices into the unsorted TP output that
    correspond to that l_out's `num_paths * (2l_out+1)` entries."""
    paths = _enumerate_paths(L1, L2, L3)
    offsets = [0]
    for l1, l2, l3 in paths:
        offsets.append(offsets[-1] + (2 * l3 + 1))
    by_l_out_indices: dict[int, list] = {l: [] for l in range(L3 + 1)}
    for k, (l1, l2, l3) in enumerate(paths):
        by_l_out_indices[l3].append(np.arange(offsets[k], offsets[k + 1]))
    return {
        l: (np.concatenate(rs).astype(np.int32) if rs else np.zeros(0, np.int32))
        for l, rs in by_l_out_indices.items()
    }


@functools.partial(jax.custom_jvp, nondiff_argnums=(1,))
def _per_l_norm(x, max_degree):
    """Per-(l, channel) L2 norm, safe gradient.

    Input shape: `[..., F, irrep_dim]`.
    Output: `[..., F, max_degree+1]`.
    """
    blocks = []
    start = 0
    for l in range(max_degree + 1):
        dim_l = 2 * l + 1
        b = jax.lax.dynamic_slice_in_dim(x, start, dim_l, axis=-1)
        blocks.append(jnp.sqrt(jnp.sum(b * b, axis=-1)))
        start += dim_l
    return jnp.stack(blocks, axis=-1)


@_per_l_norm.defjvp
def _per_l_norm_jvp(max_degree, primals, tangents):
    (x,) = primals
    (xd,) = tangents
    n = _per_l_norm(x, max_degree)
    out = []
    start = 0
    for l in range(max_degree + 1):
        dim_l = 2 * l + 1
        xb = jax.lax.dynamic_slice_in_dim(x, start, dim_l, axis=-1)
        db = jax.lax.dynamic_slice_in_dim(xd, start, dim_l, axis=-1)
        nb = jnp.expand_dims(n[..., l], axis=-1)
        safe = jnp.where(nb > 0, nb, 1.0)
        out.append(jnp.sum(db * (xb / safe), axis=-1))
        start += dim_l
    return n, jnp.stack(out, axis=-1)


def _per_l_repeat(coef, max_degree):
    """Repeat coefficient `[..., max_degree+1]` per-l over (2l+1) m-values."""
    out = []
    for l in range(max_degree + 1):
        c = coef[..., l]
        out.append(jnp.broadcast_to(c[..., None], c.shape + (2 * l + 1,)))
    return jnp.concatenate(out, axis=-1)


# -- e3x-compatible Dense and Tensor on top of e3j ----------------------------


class _DenseE3j(nn.Module):
    """Equivalent of `e3x.nn.Dense` (per-l block channel mixing, no irrep mix).

    Stores parameters in `(features_in, features_out)` per-l layout matching
    e3x. Used both as a standalone module and as the front of TensorDense.
    """

    features: int
    max_degree: int
    use_bias: bool = False

    @nn.compact
    def __call__(self, x):
        # x: [..., F_in, irrep_dim(max_degree)]
        F_in = x.shape[-2]
        out_blocks = []
        start = 0
        for l in range(self.max_degree + 1):
            dim_l = 2 * l + 1
            w_l = self.param(
                f"kernel_{l}",
                jax.nn.initializers.lecun_normal(),
                (F_in, self.features),
            )
            block = jax.lax.dynamic_slice_in_dim(x, start, dim_l, axis=-1)
            # block: [..., F_in, 2l+1]; mix features dim: [..., F_out, 2l+1]
            mixed = jnp.einsum("...fm,fF->...Fm", block, w_l)
            if self.use_bias and l == 0:
                b_l = self.param(f"bias_{l}", jax.nn.initializers.zeros, (self.features,))
                mixed = mixed + b_l[..., None]
            out_blocks.append(mixed)
            start += dim_l
        return jnp.concatenate(out_blocks, axis=-1)


class _TensorE3j(nn.Module):
    """Equivalent of `e3x.nn.Tensor` (parameter-equivalent to e3x).

    Kernel layout matches e3x exactly:
        `(1, l_in1+1, 1, l_in2+1, 1, l_out+1, features)`
    The `1`s are the parity axes (single, since we don't carry pseudotensors).
    Forbidden parity paths receive zero contribution at evaluation.
    """

    l_in1: int
    l_in2: int
    l_out: int

    @nn.compact
    def __call__(self, x, y):
        # x: [N, F, irrep_dim(l_in1)], y: [N, F, irrep_dim(l_in2)]
        L1, L2, L3 = self.l_in1, self.l_in2, self.l_out
        F = x.shape[-2]
        # e3x's default_tensor_kernel_init is lecun-style scaled by 1/sqrt(F).
        kernel = self.param(
            "kernel",
            jax.nn.initializers.normal(stddev=1.0 / np.sqrt(F)),
            (1, L1 + 1, 1, L2 + 1, 1, L3 + 1, F),
        )
        tp = _tensor_product(L1, L2, L3)
        z = tp(x, y)  # [N, F, target.dim]
        path_indices = _path_slices(L1, L2, L3)
        paths_by_l = _paths_by_l_out(L1, L2, L3)
        out_blocks = []
        for l_out in range(L3 + 1):
            paths = paths_by_l[l_out]
            if not paths:
                out_blocks.append(jnp.zeros(z.shape[:-1] + (2 * l_out + 1,), dtype=z.dtype))
                continue
            idx = jnp.asarray(path_indices[l_out])
            block = z[..., idx]
            n_paths = len(paths)
            block = block.reshape(z.shape[:-1] + (n_paths, 2 * l_out + 1))
            # CG ratio e3x/e3nn = sqrt(2*l_out+1) (verified empirically over
            # all (l1, l2, l_out) paths up to l_max=4).
            cg_ratio = np.sqrt(2 * l_out + 1)
            ws = []
            for l1, l2 in paths:
                w = kernel[0, l1, 0, l2, 0, l_out, :]  # (F,)
                ws.append(w * cg_ratio)
            w_l = jnp.stack(ws, axis=0)  # (num_paths, F)
            collapsed = jnp.einsum("...fpm,pf->...fm", block, w_l)
            out_blocks.append(collapsed)
        return jnp.concatenate(out_blocks, axis=-1)


class _TensorDenseE3j(nn.Module):
    """Equivalent of `e3x.nn.TensorDense`: `Dense(2F_out)` then `Tensor`.

    `output_max_degree` controls the output l_max (default: same as input).
    """

    features: int
    max_degree: int
    output_max_degree: int | None = None
    use_bias: bool = False

    @nn.compact
    def __call__(self, x):
        F_out = self.features
        L = self.max_degree
        L_out = L if self.output_max_degree is None else self.output_max_degree
        proj = _DenseE3j(
            features=2 * F_out, max_degree=L, use_bias=self.use_bias, name="dense"
        )(x)
        x1, x2 = jnp.split(proj, 2, axis=-2)
        return _TensorE3j(l_in1=L, l_in2=L, l_out=L_out, name="tensor")(x1, x2)


class _MessagePassE3j(nn.Module):
    """e3x.nn.MessagePass equivalent (sparse src/dst index list).

    Bundles `Dense(features=F)(basis) -> Tensor(filters, gathered_inputs)`
    so the param tree matches e3x: children are `filter` and `tensor`.
    """

    features: int
    max_degree: int

    @nn.compact
    def __call__(self, inputs, basis, dst_idx, src_idx, num_segments):
        gathered = inputs[src_idx]  # [P, F, irrep_dim]
        filters = _DenseE3j(
            features=self.features, max_degree=self.max_degree, name="filter"
        )(basis)
        products = _TensorE3j(
            l_in1=self.max_degree,
            l_in2=self.max_degree,
            l_out=self.max_degree,
            name="tensor",
        )(filters, gathered)
        return jax.ops.segment_sum(products, dst_idx, num_segments=num_segments)


# -- initial embedding --------------------------------------------------------


class InitialE3j(nn.Module):
    cutoff: float = 5.0
    max_degree: int = 4
    num_features: int = 128
    num_radial: int = 32
    num_species: int = 8
    num_spherical_features: int = 4
    cutoff_fn: str = "cosine_cutoff"
    radial_basis: str = "basic_bernstein"

    @nn.compact
    def __call__(self, R_ij, Z_i, pair_mask, atom_mask):
        cutoff_fn = getattr(e3x.nn.functions, self.cutoff_fn)
        R_hat, r_ij = normalize_and_return_norm(R_ij, axis=-1)
        R_hat *= pair_mask[..., None]
        cutoffs = cutoff_fn(r_ij, cutoff=self.cutoff) * pair_mask
        radial_expansion = (
            RadialEmbedding(self.num_radial, self.cutoff, function=self.radial_basis)(r_ij)
            * cutoffs[..., None]
        )

        # Use e3x's SH directly (no convention conversion needed downstream).
        # Output shape: [pairs, irrep_dim] in e3x's normalisation,
        # m-axis in (-l,...,+l) order (cartesian_order=False).
        spherical_expansion = e3x.so3.spherical_harmonics(
            R_hat,
            max_degree=self.max_degree,
            r_is_normalized=True,
            cartesian_order=False,
        )
        spherical_expansion *= pair_mask[..., None]

        species_expansion = (
            ChemicalEmbedding(num_features=self.num_species)(Z_i) * atom_mask[..., None]
        )

        return radial_expansion, spherical_expansion, species_expansion, cutoffs, r_ij


# -- main model ---------------------------------------------------------------


class LoremE3j(nn.Module):
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
    def __call__(self, Z_i, sr, nopbc, pbc):
        R = sr.positions
        i = sr.centers
        j = sr.others
        cell = sr.cell
        cell_shifts = sr.cell_shifts
        pair_mask = sr.pair_mask
        atom_mask = sr.atom_mask

        R_ij = (
            R[j] - R[i] + jnp.einsum("pA,pAa->pa", cell_shifts, cell[sr.pair_to_structure])
        )

        num_atoms = Z_i.shape[0]
        num_pairs = R_ij.shape[0]
        L = self.max_degree
        L_lr = self.max_degree_lr
        num_l = L + 1
        irrep_dim = (L + 1) ** 2
        d = self.num_features
        s = self.num_spherical_features
        l_factors = jnp.array([(2 * l + 1) for l in range(L + 1)], dtype=float) ** 0.25

        radial, spherical, species, cutoffs, r_ij = InitialE3j(
            cutoff=self.cutoff,
            max_degree=L,
            num_features=d,
            num_radial=self.num_radial,
            num_species=self.num_species,
            num_spherical_features=s,
            cutoff_fn=self.cutoff_fn,
            radial_basis=self.radial_basis,
        )(R_ij, Z_i, pair_mask, atom_mask)

        edges_scalar = RadialCoefficients(d)(
            jnp.concatenate([species[i], species[j]], axis=-1),
            radial,
            cutoffs,
            pair_mask,
        )

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

        # Build edges_spherical [pairs, s, irrep_dim]
        coefficients = masked(
            nn.Dense(num_l * s, use_bias=False), edges_scalar, pair_mask
        ).reshape(num_pairs, num_l, s)
        coefficients = jnp.transpose(coefficients, (0, 2, 1))  # [pairs, s, num_l]
        coefficients = _per_l_repeat(coefficients, L)  # [pairs, s, irrep_dim]
        edges_spherical = coefficients * spherical[:, None, :]

        nodes_spherical = (
            jax.ops.segment_sum(edges_spherical, i, num_segments=num_atoms)
            * atom_mask[..., None, None]
        )  # [atoms, s, irrep_dim]

        nodes_spherical = _TensorDenseE3j(features=s, max_degree=L)(nodes_spherical)

        # Norms -> scalar features
        norms = _per_l_norm(nodes_spherical, L)  # [atoms, s, num_l]
        norms = jnp.transpose(norms, (0, 2, 1))  # [atoms, num_l, s]
        updates = (norms * l_factors[None, :, None]).reshape(num_atoms, -1)
        nodes_scalar = Update(d)(nodes_scalar, updates, atom_mask)

        energy = masked(MLP(features=[d, d, 1]), nodes_scalar, atom_mask)[..., 0]

        # Message passing
        for _ in range(self.num_message_passing):
            edges_scalar = RadialCoefficients(d)(
                jnp.concatenate([nodes_scalar[i], nodes_scalar[j]], axis=-1),
                radial,
                cutoffs,
                pair_mask,
            )
            updates = (
                jax.ops.segment_sum(
                    masked(nn.Dense(d, use_bias=False), edges_scalar, pair_mask),
                    i,
                    num_segments=num_atoms,
                )
                * atom_mask[..., None]
            )
            nodes_scalar = Update(d)(nodes_scalar, updates, atom_mask)

            if self.equivariant_message_passing:
                coefficients = masked(
                    nn.Dense(num_l * s, use_bias=False), edges_scalar, pair_mask
                ).reshape(num_pairs, num_l, s)
                coefficients = jnp.transpose(coefficients, (0, 2, 1))
                coefficients = _per_l_repeat(coefficients, L)
                edges_spherical = coefficients * spherical[:, None, :]

                # Bundled MessagePass: filter Dense + Tensor(filters, gather(nodes)).
                messages = (
                    _MessagePassE3j(features=s, max_degree=L)(
                        inputs=nodes_spherical,
                        basis=edges_spherical,
                        dst_idx=i,
                        src_idx=j,
                        num_segments=num_atoms,
                    )
                    * atom_mask[..., None, None]
                )

                # Combine current node spherical with messages:
                # Tensor(Dense(node), Dense(message)) — matches e3x.
                nodes_spherical = _TensorE3j(l_in1=L, l_in2=L, l_out=L)(
                    _DenseE3j(features=s, max_degree=L)(nodes_spherical),
                    _DenseE3j(features=s, max_degree=L)(messages),
                )

                norms = _per_l_norm(nodes_spherical, L)
                norms = jnp.transpose(norms, (0, 2, 1))
                updates = (norms * l_factors[None, :, None]).reshape(num_atoms, -1)
                nodes_scalar = Update(d)(nodes_scalar, updates, atom_mask)

            energy += masked(MLP(features=[d, d, 1]), nodes_scalar, atom_mask)[..., 0]

        if self.lr:
            scalar_charges = masked(MLP(features=[2 * d, 1]), nodes_scalar, atom_mask)
            # Self-TP + mix to 1 channel; restrict output to max_degree_lr.
            full_lr = _TensorDenseE3j(features=1, max_degree=L, output_max_degree=L_lr)(
                nodes_spherical
            )
            # full_lr shape: [atoms, 1, (L_lr+1)**2]
            spherical_charges = full_lr[..., 0, :]
            charges = jnp.concatenate([scalar_charges, spherical_charges], axis=-1)

            calculator = Ewald()
            potentials = jax.vmap(
                lambda q: calculator.potentials(q, sr, nopbc, pbc),
                in_axes=-1,
                out_axes=-1,
            )(charges)
            scalar_potential = potentials[..., 0][..., None]
            spherical_potential = potentials[..., 1:]  # [atoms, dim_lr]

            # In e3x, Lorem does:
            #   Tensor(Dense(spherical_potential, features=s), nodes_spherical)
            # spherical_potential has max_degree=L_lr; nodes_spherical has L.
            sph_pot_1ch = spherical_potential[:, None, :]  # [atoms, 1, dim_lr]
            sph_pot_mixed = _DenseE3j(features=s, max_degree=L_lr)(
                sph_pot_1ch
            )  # [atoms, s, dim_lr]

            spherical_updates = _TensorE3j(l_in1=L_lr, l_in2=L, l_out=L)(
                sph_pot_mixed, nodes_spherical
            )

            norms = _per_l_norm(spherical_updates, L)
            norms = jnp.transpose(norms, (0, 2, 1))
            norms = (norms * l_factors[None, :, None]).reshape(num_atoms, -1)
            updates = jnp.concatenate([scalar_potential, norms], axis=-1)
            nodes_scalar = Update(d)(nodes_scalar, updates, atom_mask)

            energy += masked(MLP(features=[d, d, 1]), nodes_scalar, atom_mask)[..., 0]

        return energy

    # -- inference helpers ---------------------------------------------------

    def atoms_to_batch(self, atoms):
        from lorem.batching import to_batch, to_sample

        sample = to_sample(atoms, self.cutoff, energy=False, forces=False, stress=False)
        batch = to_batch([sample], [])
        return jax.tree.map(lambda x: jnp.array(x), batch)

    def dummy_inputs(self):
        from ase.build import bulk

        atoms = bulk("Ar") * [2, 2, 2]
        return self.atoms_to_batch(atoms)[:-1]

    def energy(self, params, batch):
        sr = batch[1]
        energies = self.apply(
            params, batch.atomic_numbers, batch.sr, batch.nopbc, batch.pbc
        )
        energies *= sr.atom_mask
        return jnp.sum(energies), energies

    def predict(self, params, batch, stress=False):
        sr = batch[1]
        energy_and_derivatives_fn = jax.value_and_grad(
            self.energy, allow_int=True, has_aux=True, argnums=1
        )
        bea, grads = energy_and_derivatives_fn(params, batch)
        _, energies = bea
        grads = grads.sr
        energy = (
            jax.ops.segment_sum(energies, sr.atom_to_structure, sr.cell.shape[0])
            * sr.structure_mask
        )
        forces = -grads.positions
        results = {"energy": energy, "forces": forces}
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
