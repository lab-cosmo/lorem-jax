"""LOREM driver for i-PI."""

from __future__ import annotations

from ipi.pes.ase import ASEDriver

from lorem import LOREMCalculator

__DRIVER_NAME__ = "lorem"
__DRIVER_CLASS__ = "LOREM_driver"


class LOREM_driver(ASEDriver):
    def __init__(self, template, model_path, device="cpu", *args, **kwargs):
        super().__init__(template, *args, **kwargs)

        self.model_path = model_path
        self.device = device
        self.capabilities.append("BEC")

    def check_parameters(self):
        super().check_parameters()
        self.ase_calculator = LOREMCalculator.from_checkpoint(self.model_path)

    def compute_structure(self, cell, pos):
        pot_ipi, force_ipi, vir_ipi, extras = super().compute_structure(cell, pos)

        if "BEC" not in extras and "apt" in extras:
            extras["BEC"] = extras["apt"]

        return pot_ipi, force_ipi, vir_ipi, extras
