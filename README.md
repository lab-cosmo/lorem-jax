# LOREM-JAX

JAX implementation of [LOREM](https://arxiv.org/abs/2504.20462) (Learning Long-Range Representations with Equivariant Messages), a machine learning interatomic potential with equivariant long-range message passing.

Built on [JAX](https://github.com/jax-ml/jax), [Flax](https://github.com/google/flax), [e3x](https://github.com/google-research/e3x), and [jax-pme](https://github.com/lab-cosmo/jax-pme).

## Installation

Requires Python >= 3.11.

```bash
pip install .
```

The package depends on `marathon` and `comms` for training. These are not yet publicly released and must be installed separately. The core model can be imported without them.

## Usage

### ASE calculator

```python
import jax
from ase.build import bulk
from lorem.models.mlip import Lorem
from lorem.calculator import Calculator

model = Lorem(cutoff=5.0)
params = model.init(jax.random.key(42), *model.dummy_inputs())
calc = Calculator.from_model(model, params=params)

atoms = bulk("Ar") * [2, 2, 2]
calc.calculate(atoms)
print(calc.results["energy"], calc.results["forces"].shape)
```

To load a trained model from a checkpoint:

```python
calc = Calculator.from_checkpoint("path/to/checkpoint")
```

### Training

Training uses `lorem-train`, which reads `model.yaml` and `settings.yaml` from the current directory:

```bash
cd my_experiment
lorem-train
```

See `examples/train-mlp/` for a complete example including data preparation and configuration files.

### Model variants

- **`Lorem`** -- the standard MLIP model (energy + forces + stress)
- **`LoremBEC`** -- predicts Born effective charges in addition to energy/forces

### Key hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `cutoff` | 5.0 | Short-range cutoff radius (A) |
| `max_degree` | 6 | Maximum angular momentum for spherical features |
| `max_degree_lr` | 2 | Maximum angular momentum for long-range charges |
| `num_features` | 128 | Number of scalar features |
| `num_spherical_features` | 8 | Number of spherical feature channels |
| `num_radial` | 32 | Number of radial basis functions |
| `num_message_passing` | 0 | Number of short-range message passing steps |
| `lr` | True | Enable long-range (Ewald) interaction |

## Installing the i-PI driver

After installation of the package, install the i-PI driver via:

```bash
lorem-install-ipi-driver
```

This copies the LOREM driver into the i-PI `pes` directory. You can rerun `lorem-install-ipi-driver` anytime (it is idempotent) if you switch environments or reinstall i-PI.

## Development

Format and lint:

```bash
ruff format . && ruff check --fix .
```

Run tests:

```bash
python -m pytest tests/ -v --override-ini="addopts="
```

## License

BSD-3-Clause
