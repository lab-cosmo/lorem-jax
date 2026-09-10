# Training example: the work function

Trains `LoremWF` to predict the work function Φ alongside energy and forces,
once per head, so the two can be compared on identical data and loss weights.

## The two heads

`LoremWF` conditions on the total charge `q` exactly as `Lorem` does (FiLM on
the invariant node features), and adds Φ as an output.
`work_function_from_energy` picks how:

- **`true`** — Φ = ∂E/∂q, read off the same backward pass that already
  produces the forces. It costs nothing extra and is consistent with the
  model's own E(q) by construction, but it can only be trained where `q`
  actually varies.
- **`false`** — a separate readout that mean-pools the invariant node
  features of every stage and passes them through an MLP, the way CP-MACE
  predicts the Fermi level (Wang et al., *J. Chem. Theory Comput.* **21**,
  7628 (2025), eq. 8). Free to fit the label, and under no obligation to agree
  with the model's own ∂E/∂q.

The convention is Φ = ∂E/∂q, with `q` the total charge in units of +e and `E`
the total energy — not the grand potential. The textbook Φ = −E_F carries a
minus because E_F = ∂E/∂N_e counts electrons; `q` counts the holes, so
∂N_e = −∂q and the two minuses cancel. There is no sign flip in the code, and
`razor`'s labels confirm it: `work_function` regresses on the dataset's own
finite-difference `dEdq_fd` at slope +0.993, r = +0.998.

Datasets have to supply `work_function` labels in this convention.

Only the energy sum is differentiated, so the direct head is a pure
side-output: it cannot perturb energy, forces or stress.

## Data

`data.xyz` is 24 frames from the Pt(111)/water slabs of
`razor_centre_paper_test.xyz` (Bergmann, Reuter & Hörmann, *J. Chem. Phys.*
**164**, 174110 (2026); GPAW, PBE-D3(BJ), Solvated Jellium Method), stratified
across the charge range so the crop still spans the response. 108 atoms,
`pbc="T T F"`, `bias_charge ∈ [−1, 1] e`, `work_function ∈ 2.1–7.1 V`.

`prepare.py` renames `bias_charge` → `total_charge`. **That rename is not
cosmetic.** `marathon.grain.prepare()` writes NaN for any declared
`atoms.info` key a frame does not carry, and because `total_charge` is a model
*input* rather than a label, the NaN reaches the network and takes the loss
down from step 0 — with no error, just `loss became NaN` several thousand
steps in. Two production runs were lost to exactly this.

Both `total_charge` and `work_function` must be declared in `PROPERTIES`;
`prepare()` silently drops any `atoms.info` entry that is not.

## Files

- `data.xyz` — cropped dataset in extended XYZ format
- `prepare.py` — splits into train/valid and writes marathon datasets
- `my_experiment_from_energy/`, `my_experiment_direct_head/` — identical apart
  from `work_function_from_energy`

## Running

```bash
# prepare data
DATASETS=. python prepare.py

# train, either head
cd my_experiment_from_energy
DATASETS=.. lorem-train
```

Five epochs on 18 frames is a smoke test, not a converged model.

## Using it

Φ comes through the ASE calculator as `work_function`:

```python
from lorem.calculator import Calculator

calc = Calculator.from_checkpoint("my_experiment_from_energy/run/checkpoints/R2_E+F+W")
atoms.info["total_charge"] = -0.5
atoms.calc = calc
print(atoms.get_potential_energy(), calc.get_property("work_function", atoms))
```

A `Calculator` may be reused across a charge sweep at fixed geometry: it
detects changes to `atoms.info["total_charge"]`, which the neighbor-list and
geometry caches cannot see.
