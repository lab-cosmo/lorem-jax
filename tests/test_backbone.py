import numpy as np
import jax
import jax.numpy as jnp

from lorem.models.backbone import (
    MLP,
    ChemicalEmbedding,
    Initial,
    RadialEmbedding,
    degree_wise_repeat,
    degree_wise_repeat_last_axis,
    degree_wise_trace,
    spherical_norm,
    spherical_norm_last_axis,
)

# -- spherical helpers --


def test_degree_wise_trace():
    max_degree = 2
    num_lm = (max_degree + 1) ** 2  # 9
    batch = 4
    x = jnp.ones((batch, num_lm))
    result = degree_wise_trace(x, max_degree)
    # l=0: 1 element, l=1: 3 elements, l=2: 5 elements
    # trace sums within each l block: [1, 3, 5]
    assert result.shape == (batch, max_degree + 1)
    np.testing.assert_allclose(result[0], [1.0, 3.0, 5.0])


def test_degree_wise_repeat():
    max_degree = 2
    x = jnp.array([10.0, 20.0, 30.0])  # one value per l
    result = degree_wise_repeat(x, max_degree, axis=-1)
    # l=0: 1 copy, l=1: 3 copies, l=2: 5 copies
    expected = jnp.array([10, 20, 20, 20, 30, 30, 30, 30, 30], dtype=float)
    assert result.shape == (9,)
    np.testing.assert_allclose(result, expected)


def test_degree_wise_repeat_last_axis():
    max_degree = 1
    num_l = max_degree + 1  # 2
    features = 3
    x = jnp.ones((5, num_l, features))
    result = degree_wise_repeat_last_axis(x, max_degree)
    # l=0: 1 copy, l=1: 3 copies -> total 4 entries on axis=1
    assert result.shape == (5, 4, features)


def test_spherical_norm():
    max_degree = 2
    num_lm = (max_degree + 1) ** 2
    # degree_wise_trace vmaps over first axis, so input must be 2D
    batch = 3
    x = jnp.ones((batch, num_lm))
    result = spherical_norm(x, max_degree)
    assert result.shape == (batch, max_degree + 1)
    # l=0: sqrt(1) = 1, l=1: sqrt(3), l=2: sqrt(5)
    np.testing.assert_allclose(result[0], jnp.sqrt(jnp.array([1.0, 3.0, 5.0])))


def test_spherical_norm_gradient():
    """Verify the custom JVP is consistent with finite differences."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        max_degree = 2
        num_lm = (max_degree + 1) ** 2
        key = jax.random.key(0)
        x = jax.random.normal(key, (1, num_lm), dtype=jnp.float64)

        def f(x):
            return spherical_norm(x, max_degree).sum()

        grad_custom = jax.grad(f)(x)

        eps = 1e-6
        grad_fd = np.zeros_like(x)
        for i in range(num_lm):
            x_plus = x.at[0, i].add(eps)
            x_minus = x.at[0, i].add(-eps)
            grad_fd[0, i] = (f(x_plus) - f(x_minus)) / (2 * eps)

        np.testing.assert_allclose(grad_custom, grad_fd, atol=1e-5)
    finally:
        jax.config.update("jax_enable_x64", prev)


def test_spherical_norm_last_axis():
    max_degree = 1
    num_lm = (max_degree + 1) ** 2  # 4
    batch, parity, features = 3, 1, 2
    X = jnp.ones((batch, parity, num_lm, features))
    result = spherical_norm_last_axis(X, max_degree)
    assert result.shape == (batch, parity, max_degree + 1, features)


# -- flax modules (no marathon needed) --


def test_mlp_output_shape():
    model = MLP(features=[32, 16, 1])
    key = jax.random.key(42)
    x = jnp.ones((5, 8))
    params = model.init(key, x)
    y = model.apply(params, x)
    assert y.shape == (5, 1)


def test_mlp_no_bias():
    model = MLP(features=[4], use_bias=False)
    key = jax.random.key(0)
    x = jnp.zeros((2, 3))
    params = model.init(key, x)
    y = model.apply(params, x)
    # zero input + no bias -> zero output
    np.testing.assert_allclose(y, 0.0, atol=1e-7)


def test_chemical_embedding():
    model = ChemicalEmbedding(num_features=8)
    key = jax.random.key(0)
    species = jnp.array([1, 6, 8])
    params = model.init(key, species)
    y = model.apply(params, species)
    assert y.shape == (3, 8)
    # different species -> different embeddings
    assert not jnp.allclose(y[0], y[1])


def test_radial_embedding():
    model = RadialEmbedding(num_features=16, cutoff=5.0, function="basic_bernstein")
    key = jax.random.key(0)
    r = jnp.array([1.0, 2.5, 4.9])
    params = model.init(key, r)
    y = model.apply(params, r)
    assert y.shape == (3, 16)
    # Bernstein basis is non-negative on [0, cutoff]
    assert float(y.min()) >= 0.0
    # different distances should give different expansions
    assert not jnp.allclose(y[0], y[1])


def test_initial_module():
    model = Initial(
        cutoff=5.0,
        max_degree=2,
        num_features=8,
        num_radial=4,
        num_species=4,
        num_spherical_features=2,
    )
    key = jax.random.key(0)
    num_pairs, num_atoms = 6, 3
    R_ij = jax.random.normal(key, (num_pairs, 3))
    Z_i = jnp.array([1, 6, 8])
    pair_mask = jnp.ones(num_pairs)
    atom_mask = jnp.ones(num_atoms)

    params = model.init(key, R_ij, Z_i, pair_mask, atom_mask)
    radial, spherical, species, cutoffs, r_ij = model.apply(
        params, R_ij, Z_i, pair_mask, atom_mask
    )

    num_lm = (2 + 1) ** 2
    assert radial.shape == (num_pairs, 4)
    assert spherical.shape == (num_pairs, num_lm)
    assert species.shape == (num_atoms, 4)
    assert cutoffs.shape == (num_pairs,)
    assert r_ij.shape == (num_pairs,)

    # masked pair should zero out
    pair_mask_partial = pair_mask.at[0].set(0)
    _, _, _, cutoffs_masked, _ = model.apply(
        params, R_ij, Z_i, pair_mask_partial, atom_mask
    )
    assert float(cutoffs_masked[0]) == 0.0
