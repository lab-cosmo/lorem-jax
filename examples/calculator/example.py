"""Example: using the LOREM ASE calculator with randomly initialised weights."""

import jax
from ase.build import bulk, molecule

from lorem.models.mlip import Lorem
from lorem.calculator import Calculator

# Create a Lorem model with default hyperparameters
model = Lorem(cutoff=5.0)

# Initialize with random parameters
key = jax.random.key(42)
params = model.init(key, *model.dummy_inputs())

# Create calculator (no baseline offset with random weights)
calc = Calculator.from_model(model, params=params)

# -- periodic bulk system --
atoms = bulk("Ar") * [2, 2, 2]
calc.calculate(atoms)

energy = calc.results["energy"]
forces = calc.results["forces"]

print("=== Periodic bulk Ar (2x2x2) ===")
print(f"Number of atoms: {len(atoms)}")
print(f"Energy: {energy:.6f} eV")
print(f"Forces shape: {forces.shape}")
print(f"Max force component: {abs(forces).max():.6f} eV/A")

# -- non-periodic molecule --
atoms_mol = molecule("H2O")
atoms_mol.center(vacuum=5.0)

calc_mol = Calculator.from_model(model, params=params)
calc_mol.calculate(atoms_mol)

print("\n=== Non-periodic H2O molecule ===")
print(f"Number of atoms: {len(atoms_mol)}")
print(f"Energy: {calc_mol.results['energy']:.6f} eV")
print(f"Forces:\n{calc_mol.results['forces']}")

# -- using ASE interface --
atoms.calc = Calculator.from_model(model, params=params)
print(f"\n=== ASE interface ===")
print(f"get_potential_energy: {atoms.get_potential_energy():.6f} eV")
print(f"get_forces max: {abs(atoms.get_forces()).max():.6f} eV/A")
