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
