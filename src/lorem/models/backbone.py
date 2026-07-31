import numpy as np
import jax
import jax.numpy as jnp

from collections.abc import Sequence

import e3x
import flax.linen as nn
from flax.core import FrozenDict
from marathon.utils import masked


def _masked(fn, x, mask):
    """Apply fn only where mask is True."""
    return masked(fn, x, mask)


# -- initial embeddings --


class Initial(nn.Module):
    cutoff: float = 5.0
    max_degree: int = 4
    num_features: int = 128
    num_radial: int = 32
    num_species: int = 8
    num_spherical_features: int = 4
    cutoff_fn: str = "cosine_cutoff"
    radial_basis: str = "basic_bernstein"

    @nn.compact
    def __call__(
        self,
        R_ij,
        Z_i,
        pair_mask,
        atom_mask,
    ):
        cutoff_fn = getattr(e3x.nn.functions, self.cutoff_fn)

        R_ij, r_ij = e3x.ops.normalize_and_return_norm(R_ij, axis=-1)
        R_ij *= pair_mask[..., None]

        cutoffs = cutoff_fn(r_ij, cutoff=self.cutoff) * pair_mask  # -> [pairs]

        radial_expansion = (
            RadialEmbedding(
                self.num_radial,
                self.cutoff,
                function=self.radial_basis,
            )(r_ij)
            * cutoffs[..., None]
        )

        spherical_expansion = e3x.so3.spherical_harmonics(
            R_ij, self.max_degree, r_is_normalized=True
        )
        spherical_expansion *= pair_mask[..., None]

        species_expansion = (
            ChemicalEmbedding(num_features=self.num_species)(Z_i) * atom_mask[..., None]
        )

        return (
            radial_expansion,
            spherical_expansion,
            species_expansion,
            cutoffs,
            r_ij,
        )


class ChemicalEmbedding(nn.Module):
    num_features: int
    total_species: int = 100

    @nn.compact
    def __call__(self, species):
        return nn.Embed(num_embeddings=self.total_species, features=self.num_features)(
            species
        )


class RadialEmbedding(nn.Module):
    num_features: int
    cutoff: int
    function: str = "basic_gaussian"
    args: FrozenDict = FrozenDict({})
    learned_transform: bool = False

    @nn.compact
    def __call__(self, r):
        function = getattr(e3x.nn.functions, self.function)

        expansion = function(
            r, **{"limit": self.cutoff, "num": self.num_features, **self.args}
        )

        if self.learned_transform:
            expansion = nn.Dense(features=self.num_features, use_bias=False)(expansion)

        return expansion


# -- basic modules --


class MLP(nn.Module):
    features: Sequence[int]
    activation: str = "silu"
    use_bias: bool = True

    @nn.compact
    def __call__(self, x):
        activation = getattr(jax.nn, self.activation)
        num_layers = len(self.features)

        for i, f in enumerate(self.features):
            x = nn.Dense(features=f, use_bias=self.use_bias)(x)
            if i != num_layers - 1:
                x = activation(x)

        return x


class Update(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x, y, atom_mask):
        x += _masked(
            MLP(features=[2 * self.features, self.features]),
            y,
            atom_mask,
        )
        x = _masked(nn.LayerNorm(), x, atom_mask)
        x += _masked(MLP(features=[2 * self.features, self.features]), x, atom_mask)
        x = _masked(nn.LayerNorm(), x, atom_mask)

        return x


# -- other modules --


class RadialCoefficients(nn.Module):
    features: int

    @nn.compact
    def __call__(self, pair_features, radial_expansion, cutoffs, pair_mask):
        num_radial = radial_expansion.shape[-1]

        coefficients = _masked(
            MLP(
                features=[
                    self.features,
                    num_radial * self.features,
                ]
            ),
            pair_features,
            pair_mask,
        )
        coefficients = coefficients.reshape(-1, num_radial, self.features)
        coefficients = jnp.einsum("prf,pr->pf", coefficients, radial_expansion)

        return coefficients


# -- helpers to deal with spherical features --


def degree_wise_trace(
    x,
    max_degree,
):
    segments = np.concatenate(
        [np.array([l] * (2 * l + 1)) for l in range(max_degree + 1)]
    ).reshape(-1)

    return jax.vmap(
        lambda _x: jax.ops.segment_sum(_x, segments, num_segments=(max_degree + 1)),
    )(x)


def degree_wise_repeat(x, max_degree, axis):
    repeats = np.array([2 * l + 1 for l in range(max_degree + 1)])

    return jnp.repeat(x, repeats, total_repeat_length=repeats.sum(), axis=axis)


def degree_wise_repeat_last_axis(x, max_degree: int):
    return jax.vmap(
        lambda y: degree_wise_repeat(y, max_degree, -1),
        in_axes=-1,
        out_axes=-1,
    )(x)


# x/||x|| (the gradient of a vector norm) is genuinely non-smooth at x=0 --
# no jnp.where-based guarding fixes a *second* derivative through a real
# singularity (needed wherever training differentiates through this twice,
# e.g. forces -> loss -> grad wrt params). Regularizing with a small eps
# makes norm (and all its derivatives) smooth everywhere, including at
# x=0, so plain autodiff handles arbitrary differentiation order without
# any custom_jvp at all.
_SPHERICAL_NORM_EPS = 1e-12


def spherical_norm(X, max_degree):
    squared = jax.lax.square(X)
    trace = degree_wise_trace(squared, max_degree)
    return jnp.sqrt(trace + _SPHERICAL_NORM_EPS)


def spherical_norm_last_axis(X, max_degree):
    # X is a e3x-style array, i.e. [batch, 1|2, lm, features]:
    # we vmap over parity and feature dimensions
    return jax.vmap(
        lambda z: jax.vmap(
            lambda x: spherical_norm(x, max_degree),
            in_axes=-1,
            out_axes=-1,
        )(z),
        in_axes=1,
        out_axes=1,
    )(X)


# -- charge conditioning --


def segment_softmax(logits, segment_ids, mask, num_segments):
    # softmax within each segment; masked entries get weight 0, and
    # segments with no unmasked entries (padding) get all-zero, not NaN
    neg_inf = jnp.finfo(logits.dtype).min
    masked_logits = jnp.where(mask, logits, neg_inf)

    segment_maxes = jax.ops.segment_max(masked_logits, segment_ids, num_segments=num_segments)
    segment_maxes = jnp.where(jnp.isfinite(segment_maxes), segment_maxes, 0.0)

    weights = jnp.where(mask, jnp.exp(masked_logits - segment_maxes[segment_ids]), 0.0)
    denom = jax.ops.segment_sum(weights, segment_ids, num_segments=num_segments)
    denom = jnp.where(denom > 0, denom, 1.0)

    return weights / denom[segment_ids]


def conserving_redistribution(values, target, weights, segment_ids, mask, num_segments):
    # shift `values` per segment so segment_sum(result) == target, using
    # non-negative `weights` to decide how the correction is distributed
    weights = weights * mask
    current = jax.ops.segment_sum(values * mask, segment_ids, num_segments=num_segments)
    excess = target - current

    weight_sum = jax.ops.segment_sum(weights, segment_ids, num_segments=num_segments)
    weight_sum = jnp.where(weight_sum > 0, weight_sum, 1.0)

    correction = weights * (excess / weight_sum)[segment_ids]
    return (values + correction) * mask


class ChargeFiLM(nn.Module):
    # FiLM conditioning of invariant node features on the per-atom charge Q_i
    features: int

    @nn.compact
    def __call__(self, Q_i, x, atom_mask):
        gamma_beta = _masked(
            MLP(features=[self.features, 2 * self.features]),
            Q_i[..., None],
            atom_mask,
        )
        gamma, beta = jnp.split(gamma_beta, 2, axis=-1)
        return (1.0 + gamma) * x + beta  # near-identity at init


class ChargeInit(nn.Module):
    # splits Q onto atoms: c_i = Q * softmax_i(MLP(h_i)), sums to Q by construction
    features: int

    @nn.compact
    def __call__(self, h, Q, atom_to_structure, atom_mask, num_structures):
        logits = _masked(nn.Dense(1), h, atom_mask)[..., 0]
        weights = segment_softmax(logits, atom_to_structure, atom_mask, num_structures)
        return weights * Q[atom_to_structure] * atom_mask


class ChargeUpdate(nn.Module):
    # one message-pass + conserving-redistribution round for the charge channel
    features: int

    @nn.compact
    def __call__(
        self,
        c,
        h,
        edges_scalar,
        i,
        pair_mask,
        atom_mask,
        Q,
        atom_to_structure,
        num_atoms,
        num_structures,
    ):
        messages = (
            jax.ops.segment_sum(
                _masked(nn.Dense(1, use_bias=False), edges_scalar, pair_mask),
                i,
                num_segments=num_atoms,
            )[..., 0]
            * atom_mask
        )
        c_tilde = (c + messages) * atom_mask

        weights = jax.nn.softplus(_masked(nn.Dense(1), h, atom_mask)[..., 0]) * atom_mask

        return conserving_redistribution(
            c_tilde, Q, weights, atom_to_structure, atom_mask, num_structures
        )
