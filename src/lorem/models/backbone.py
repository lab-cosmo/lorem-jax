import numpy as np
import jax
import jax.numpy as jnp

import functools
from collections.abc import Sequence

import e3x
import flax.linen as nn
from flax.core import FrozenDict


def _masked(fn, x, mask):
    """Apply fn only where mask is True. Lazy import from marathon."""
    from marathon.utils import masked

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


@functools.partial(jax.custom_jvp, nondiff_argnums=(1,))
def spherical_norm(X, max_degree):
    squared = jax.lax.square(X)
    trace = degree_wise_trace(squared, max_degree)
    norm = jnp.sqrt(trace)
    return norm


@spherical_norm.defjvp
def spherical_norm_jvp(max_degree, primals, tangents):
    (x,) = primals
    (x_dot,) = tangents
    primal_out = spherical_norm(x, max_degree)

    x_hat = x / degree_wise_repeat(jnp.where(primal_out > 0, primal_out, 1), max_degree, -1)

    tangent_out = degree_wise_trace(x_dot * x_hat, max_degree)
    return primal_out, tangent_out


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
