# Training example: charge conditioning

Trains `Lorem` on a small mixed-charge-state dataset, to check that
conditioning the model on the total charge `Q` of a structure actually helps
it tell charge states apart. `Lorem`/`LoremBEC` always condition on `Q` via
FiLM (see `backbone.py`'s `ChargeEmbedding`) — there's no config knob to turn this
off, since a missing/zero `Q` is harmless (the FiLM layer is a near-identity
transform at `Q=0`).

## Data

`data.xyz` is a cropped slice (60 structures, stratified across the two
charge states) of the Ag₃⁺/Ag₃⁻ dataset from Ko, Finkler, Goedecker &
Behler, *Nat. Commun.* **12**, 398 (2021) — see
`~/projects/lorem-q/Ag_clusters/README.md` for the full dataset and
provenance. Every Ag₃ trimer here is small enough that the whole cluster
sits inside any reasonable cutoff, so a purely local model has no
structural excuse for failing to distinguish the two charge states — this
isolates the value of Q-conditioning itself, independent of any
long-range/beyond-cutoff effects. That's also why `lr: false` in
`model.yaml`: this dataset isn't the place to test the Ewald long-range
channel.

The original `tot_charge` field in `atoms.info` has been renamed to
`total_charge`, which is the key `lorem/batching.py` reads to populate the
model's `Q` input. `Lorem`/`LoremBEC` default to `charge_conditioning="film"`
and treat a missing `total_charge` as Q=0, so this is also the ordinary
(unconditioned) training path if you ever want to compare against
`charge_conditioning="none"`.

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
