"""Example: evaluating a trained checkpoint with the ASE calculator.

No need to write a custom batched evaluation pipeline for casual use --
`Calculator.from_checkpoint` loads a trained model directly from a
checkpoint folder (model + params + per-species baseline), and from there
it's a normal ASE calculator: attach it to `atoms` and call
`get_potential_energy()`/`get_forces()`.

Run this after training the `train-mlp` example:

    cd ../train-mlp
    DATASETS=. python prepare.py
    cd my_experiment && DATASETS=.. lorem-train
    cd ../../eval && python example.py
"""

from pathlib import Path

from ase.build import molecule

from lorem.calculator import Calculator

CHECKPOINT = Path("../train-mlp/my_experiment/run/checkpoints/R2_E+F")

calc = Calculator.from_checkpoint(CHECKPOINT)

atoms = molecule("H2O")
atoms.calc = calc

print(f"energy: {atoms.get_potential_energy():.6f} eV")
print(f"forces:\n{atoms.get_forces()}")

# -- conditioning inputs (total_charge, external_field) --
#
# These come from atoms.info, exactly like they do during training, and
# default to zero if not set. Setting them BEFORE the first calculate()
# call on a given atoms object works as expected:
atoms.info["total_charge"] = -1.0
atoms.info["external_field"] = [0.1, 0.0, 0.0]
print(f"\ncharged + field energy: {atoms.get_potential_energy():.6f} eV")

# Gotcha: if you REUSE the same Calculator on the same atoms object and only
# change atoms.info (e.g. sweeping external_field at fixed geometry), the
# Calculator correctly detects that too -- it doesn't just cache the first
# total_charge/external_field it ever saw. This works because Calculator.update()
# checks atoms.info in addition to positions/cell, not despite reusing the
# same instance:
for field_x in (-0.2, -0.1, 0.0, 0.1, 0.2):
    atoms.info["external_field"] = [field_x, 0.0, 0.0]
    print(f"  field_x={field_x:+.1f}: E={atoms.get_potential_energy():.6f} eV")
