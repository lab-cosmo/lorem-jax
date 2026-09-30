import jax
import jax.numpy as jnp

from lorem.models.bec import PerParticleTensorPredictor


def test_tensor_predictor_chains_layers():
    x = jax.random.normal(jax.random.key(0), (5, 1, 9, 4))
    head = PerParticleTensorPredictor(features=16)
    params = head.init(jax.random.key(1), x)

    assert params["params"]["Dense_1"]["0+"]["kernel"].shape == (16, 16)

    # every Dense layer must influence the output
    for name in ["Dense_0", "Dense_1"]:
        perturbed = jax.tree.map(lambda p: p, params)
        perturbed["params"][name] = jax.tree.map(
            lambda p: p + 1.0, perturbed["params"][name]
        )
        assert not jnp.allclose(head.apply(params, x), head.apply(perturbed, x))
