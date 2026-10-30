# LOREM-JAX

JAX implementation of [LOREM](https://openreview.net/forum?id=pZI9e4SW9P) (Learning Long-Range Representations with Equivariant Messages), a machine learning interatomic potential with equivariant long-range message passing.

Built on [JAX](https://github.com/jax-ml/jax), [Flax](https://github.com/google/flax), [e3x](https://github.com/google-research/e3x), and [jax-pme](https://github.com/lab-cosmo/jax-pme).

## Installation

Requires Python >= 3.11.

```bash
pip install .
```

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

Training a model involves three steps: preparing the data, configuring the model and training settings, and running the training script.

#### 1. Prepare data

Training data is stored in [marathon](https://github.com/sirmarcel/marathon) format. Convert your extended XYZ dataset using a preparation script (see `examples/train-mlp/prepare.py` for a template):

```python
from marathon.data import datasets, get_splits
from marathon.grain import prepare

# datasets is a Path resolved from the $DATASETS environment variable
prepare(train_atoms, folder=datasets / "my_project/train", ...)
prepare(valid_atoms, folder=datasets / "my_project/valid", ...)
```

The `$DATASETS` environment variable sets the root directory where prepared datasets are stored. All dataset paths in `settings.yaml` are resolved relative to this directory.

#### 2. Configure the experiment

Each experiment lives in its own directory containing two YAML files:

**`model.yaml`** defines the model architecture:

```yaml
model:
  lorem.Lorem:
    cutoff: 5.0
    max_degree: 4
    max_degree_lr: 2
    num_features: 128
    num_spherical_features: 4
    num_message_passing: 1
```

Use `lorem.LoremBEC` instead of `lorem.Lorem` to train a model that additionally predicts Born effective charges.

**`settings.yaml`** configures training:

```yaml
train: "my_project/train"           # path relative to $DATASETS
valid: "my_project/valid"           # path relative to $DATASETS
seed: 23
batcher:
  batch_size: 4
loss_weights: {"energy": 0.5, "forces": 0.5}
optimizer: adam                      # adam or muon
start_learning_rate: 1e-3
min_learning_rate: 1e-6
max_epochs: 2000
valid_every_epoch: 2
decay_style: linear                  # linear, exponential, or warmup_cosine
use_wandb: True
```

<details>
<summary>All training settings</summary>

| Setting | Default | Description |
|---|---|---|
| `train` | *required* | Training dataset path (relative to `$DATASETS`) |
| `valid` | *required* | Validation dataset path (relative to `$DATASETS`) |
| `test_datasets` | `{}` | Extra test datasets: `{name: [path, save_predictions]}` |
| `batcher.batch_size` | *required* | Samples per batch |
| `batcher.size_strategy` | `powers_of_4` | Padding strategy for batch dimensions |
| `batcher.k_strategy`, `batcher.lr_pairs_strategy` | the kernel's | Padding strategy for k-vectors and long-range pairs |
| `batcher.max_total_*` | unbounded | Caps on the padded batch arrays: `atoms`, `pairs`, `lr_pairs`, `nobpc_pairs`, `atoms_pbc`, `k` |
| `batcher.shapes` | none | Fixed shapes for every batch, see [Fixed batch shapes](#fixed-batch-shapes) |
| `lr_kernel` | `Ewald` at the model cutoff | Long-range kernel for training, see [Long-range kernels](#long-range-kernels) |
| `loss_weights` | `{"energy": 0.5, "forces": 0.5}` | Per-target loss weights |
| `scale_by_variance` | `False` | Scale loss weights by validation set variance |
| `optimizer` | `adam` | Optimizer (`adam`, `muon`, or any optax optimizer) |
| `start_learning_rate` | `1e-3` | Initial learning rate |
| `min_learning_rate` | `1e-6` | Minimum learning rate |
| `max_epochs` | `2000` | Maximum training epochs |
| `valid_every_epoch` | `2` | Validate every N epochs |
| `decay_style` | `linear` | LR schedule: `linear`, `exponential`, or `warmup_cosine` |
| `start_decay_after` | `10` | Epoch to begin LR decay |
| `stop_decay_after` | `max_epochs` | Epoch to end LR decay (linear only) |
| `warmup_epochs` | `0` | Warmup epochs (`warmup_cosine` only) |
| `gradient_clip` | `0` | Gradient clipping threshold (0 = disabled) |
| `seed` | `0` | Random seed |
| `rotational_augmentation` | `False` | Apply random rotations to training data |
| `filter_mixed_pbc` | `False` | Filter out structures with mixed periodic boundary conditions |
| `filter_above_num_atoms` | `False` | Filter out structures above this atom count |
| `checkpointers` | `default` | `default` or `full` (adds RMSE checkpointers) |
| `use_wandb` | `True` | Log to Weights & Biases |
| `wandb_project` | auto | W&B project name (default: derived from folder names) |
| `wandb_name` | auto | W&B run name (default: experiment folder name) |
| `benchmark_pipeline` | `True` | Benchmark data pipeline before training |
| `compilation_cache` | `False` | Enable JAX persistent compilation cache |
| `default_matmul_precision` | `float32` | JAX matmul precision (`default`, `float32`) |
| `debug_nans` | `False` | Enable JAX NaN debugging (~50% slowdown) |
| `enable_x64` | `False` | Enable 64-bit floating point |
| `worker_count` | `4` | Data loading workers (training) |
| `worker_count_valid` | `worker_count` | Data loading workers (validation) |
| `worker_buffer_size` | `2` | Prefetch buffer per worker (training) |

</details>

#### 3. Run training

```bash
cd my_experiment
DATASETS=/path/to/datasets lorem-train
```

Training writes checkpoints, logs, and plots to a `run/` directory inside the experiment folder. If a `run/` directory already exists, training resumes from the latest checkpoint.

See `examples/train-mlp/` and `examples/train-bec/` for complete examples including data preparation and configuration files.

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

### Long-range kernels

The long-range part is computed by a kernel from `lorem.longrange`. A kernel has no parameters, only convergence settings, so it is not part of the model: a trained model runs with any kernel, which changes speed and convergence but not the model.

| Kernel | Description |
|---|---|
| `Ewald()` | Default. Batched Ewald (`jaxpme.batched_mixed`) with the real-space cutoff at the model cutoff |
| `Ewald(num_k=...)` | Same backend, with a fixed k-grid size; the real-space cutoff and smearing follow from it, independent of the model cutoff |
| `TiledEwald(num_k=...)` | Tile-dispatched batched Ewald (`jaxpme.batched_tiled`), for training on heterogeneous batches |

`Ewald` takes either `num_k` or `cutoff=` (the real-space cutoff), not both: with `num_k`, the smearing follows the k-grid, and a smaller cutoff would not converge the real-space sum. For training, set the kernel in `settings.yaml`:

```yaml
lr_kernel:
  TiledEwald:
    num_k: 1000
```

For inference, `Calculator.from_checkpoint` uses the kernel the model was trained with, and `Ewald()` for checkpoints that did not record one. Pass `lr_kernel=` to use another, e.g. `Calculator.from_checkpoint(folder, lr_kernel=Ewald(num_k=1000))`.

With a fixed `num_k`, the real-space cutoff grows with the cell, and so do the long-range pairs: for large cells, they and the k-grid are the largest arrays in a batch. Each kernel pads them to its own strategy (`TiledEwald` to whole tiles); `batcher.k_strategy` and `batcher.lr_pairs_strategy` override it.

### Fixed batch shapes

By default, each batch is padded to what it holds, so a run compiles once per distinct shape. With `batcher.shapes`, every batch has exactly the given shapes and a run compiles once; samples join a batch while it has room, and a sample that fits no batch raises:

```yaml
batcher:
  batch_size: 8          # structures, one of them padding
  shapes:
    atoms: 4096          # sum over structures, plus a padding atom
    pairs: 131072        # short-range pairs, plus a padding pair
    lr_pairs: 4194304    # Ewald real-space pairs, plus a padding pair
    pairs_nonpbc: 1      # all-pairs list of non-periodic structures
    pbc: 4               # periodic structures
    atoms_pbc: 4096      # Ewald: atoms of the largest periodic structure; TiledEwald: all of them, in whole BM tiles
    k: 1152              # k-vectors of the largest grid; TiledEwald: a multiple of BK
```

## Installing the i-PI driver

After installation of the package, install the i-PI driver via:

```bash
lorem-install-ipi-driver
```

This copies the LOREM driver into the i-PI `pes` directory. You can rerun `lorem-install-ipi-driver` anytime (it is idempotent) if you switch environments or reinstall i-PI.

## Exporting checkpoints

To use a trained model outside JAX (e.g. with the torch port in [metatrain](https://github.com/metatensor/metatrain)), export a checkpoint with the script in `scripts/`. The script itself needs JAX and flax, but neither lorem-jax nor marathon:

```bash
uv run path/to/lorem-jax/scripts/export.py run/checkpoints/R2_E+F exported/
```

This writes `params.npz` (flax parameters keyed by `/`-joined paths, e.g. `Initial_0/ChemicalEmbedding_0/Embed_0/embedding`), the checkpoint's `model.yaml` and `baseline.yaml`, and `export.yaml` with the export format version. Parameters are not reshaped for any particular consumer. Ewald convergence settings are not part of the model and are not exported.

## Development

Format and lint:

```bash
ruff format . && ruff check --fix .
```

Run tests:

```bash
python -m pytest tests/ -v --override-ini="addopts="
```

Or use tox:

```bash
tox -e lint       # check formatting + linting
tox -e tests      # run unit tests
tox -e examples   # run examples as smoke tests
tox -e format     # auto-format
```

## License

BSD-3-Clause


This project is [maintained](https://github.com/lab-cosmo/.github/blob/main/Maintainers.md) by [@PicoCentauri](https://github.com/PicoCentauri) and [@sirmarcel](https://github.com/sirmarcel), who will reply to issues and pull requests opened on this repository as soon as possible. You can mention them directly if you did not receive an answer after a couple of days.
