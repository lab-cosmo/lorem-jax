"""LOREM driver for i-PI."""

from __future__ import annotations

from ipi.pes.ase import ASEDriver

from lorem.calculator import Calculator

__DRIVER_NAME__ = "lorem"
__DRIVER_CLASS__ = "LOREM_driver"


class LOREM_driver(ASEDriver):
    def __init__(self, template, model_path, *args, **kwargs):
        self.model_path = model_path
        super().__init__(template, *args, **kwargs)
        self.capabilities.append("born_effective_charges")

    def check_parameters(self):
        super().check_parameters()
        self.ase_calculator = Calculator.from_checkpoint(self.model_path)

    def compute_structure(self, cell, pos):
        pot_ipi, force_ipi, vir_ipi, extras = super().compute_structure(cell, pos)

        # BEC passthrough: when model outputs "born_effective_charges" (e.g. LoremBEC),
        # expose as "BEC" in (3*natoms, 3) layout for i-PI compatibility
        if "born_effective_charges" in extras:
            extras["BEC"] = extras.pop("born_effective_charges").reshape(-1, 3)

        return pot_ipi, force_ipi, vir_ipi, extras
