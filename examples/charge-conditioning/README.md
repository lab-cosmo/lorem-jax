# Training example: charge conditioning

Trains `Lorem` on a small mixed-charge-state dataset, to check that conditioning the model on the total charge `Q` of a structure helps it tell charge states apart. With `charge_conditioning: true`, the model declares `total_charge` as an input and applies a FiLM layer (`ChargeConditioning` in `backbone.py`) to the scalar node features, which is the identity at `Q=0`.

## Data

`data.xyz` is a cropped slice (60 structures, stratified across the two charge states) of the Ag₃⁺/Ag₃⁻ dataset from Ko, Finkler, Goedecker & Behler, *Nat. Commun.* **12**, 398 (2021), <https://doi.org/10.1038/s41467-020-20427-2>, whose supplementary data carries the full set. Every Ag₃ trimer is small enough to sit inside any reasonable cutoff, so a purely local model has no structural excuse for failing to distinguish the two charge states; this isolates the value of Q-conditioning from long-range effects, hence `lr: false`.

The original `tot_charge` field in `atoms.info` has been renamed to `total_charge`. `prepare.py` declares it as a property with `storage: atoms.info`, so it is stored in the dataset and read into `batch.inputs`. Missing values become `Q=0`.

## Files

- `data.xyz` — cropped dataset in extended XYZ format
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
