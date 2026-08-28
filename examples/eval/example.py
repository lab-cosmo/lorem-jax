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

# -- conditioning input (total_charge) --
#
# This comes from atoms.info, exactly like it does during training, and
# defaults to zero if not set. Setting it BEFORE the first calculate()
# call on a given atoms object works as expected:
atoms.info["total_charge"] = -1.0
print(f"\ncharged energy: {atoms.get_potential_energy():.6f} eV")

# Gotcha: if you REUSE the same Calculator on the same atoms object and only
# change atoms.info (e.g. sweeping total_charge at fixed geometry), the
# Calculator correctly detects that too -- it doesn't just cache the first
# total_charge it ever saw. This works because Calculator.update() checks
# atoms.info in addition to positions/cell, not despite reusing the same
# instance:
for q in (-1.0, -0.5, 0.0, 0.5, 1.0):
    atoms.info["total_charge"] = q
    print(f"  total_charge={q:+.1f}: E={atoms.get_potential_energy():.6f} eV")
