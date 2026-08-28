# Evaluation example: the ASE calculator

Shows how to evaluate a trained checkpoint without writing a custom batched
evaluation pipeline -- `Calculator.from_checkpoint` loads model + params +
per-species baseline directly from a checkpoint folder, and from there it
behaves like any other ASE calculator.

## Files

- `example.py` -- loads a checkpoint, attaches it to an `ase.Atoms` object,
  and calls `get_potential_energy()`/`get_forces()`. Also demonstrates the
  `total_charge` conditioning input via `atoms.info`, including reusing one
  `Calculator` across a charge sweep at fixed geometry.

## Running

```bash
# train a small model first (see ../train-mlp/README.md)
cd ../train-mlp
DATASETS=. python prepare.py
cd my_experiment && DATASETS=.. lorem-train

cd ../../eval
python example.py
```

## `Calculator.from_model` vs. `Calculator.from_checkpoint`

- `Calculator.from_model(model, params=...)` -- wrap an in-memory model
  (e.g. freshly initialized, or already loaded some other way). See
  `../calculator/example.py`.
- `Calculator.from_checkpoint(folder)` -- load model + trained params +
  per-species baseline directly from a checkpoint folder written by
  `lorem-train` (e.g. `my_experiment/run/checkpoints/R2_E+F`).

## When NOT to use the per-structure calculator

For a handful of structures, or interactive/exploratory use, the calculator
above is the simplest path. For evaluating thousands of structures of
varying size (e.g. a held-out test set spanning many different atom
counts), calling the ASE calculator once per structure triggers a fresh XLA
compile per distinct padded shape, which gets slow and can blow up GPU
memory as the number of distinct shapes grows. In that regime, batch
structures through `model.to_sample`/`model.to_batch` directly (see the
`evaluate.py` scripts under `../../../lorem-q-work/experiments/*/` for the
pattern: sort by atom count, batch, run `model.predict` per batch).

## Conditioning input and calculator caching

`total_charge` is read from `atoms.info`, defaults to zero if unset, and --
unlike positions/cell -- isn't tracked by the neighbor-list cache.
`Calculator.update()` checks for changes to it independently, so reusing
one `Calculator` instance across a sweep of `atoms.info["total_charge"]` at
fixed geometry picks up each change correctly rather than silently reusing
the first value it saw.
