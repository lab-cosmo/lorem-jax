# Training example: MLP

Trains a `Lorem` model on a small dataset for 2 epochs.

## Files

- `data.xyz` — toy dataset in extended XYZ format
- `prepare.py` — splits data into train/valid and writes marathon datasets
- `my_experiment/model.yaml` — model configuration
- `my_experiment/settings.yaml` — training settings

## Running

```bash
# prepare data
DATASETS=. python prepare.py

# run training
cd my_experiment
DATASETS=.. lorem-train
```

## Fine-tuning from pretrained weights

The `my_experiment_finetune/` directory demonstrates restarting training from
pretrained model weights. Add `initial_weights` to `settings.yaml`:

```yaml
initial_weights: "path/to/previous_run/checkpoints/R2_E+F/model/model.msgpack"
```

Only model weights are loaded — optimizer state, step counter, and data iterator
start fresh. If the current model has layers not present in the source checkpoint,
those layers keep their random initialization.
