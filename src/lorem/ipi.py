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
        self.capabilities.append("BEC")

    def check_parameters(self):
        super().check_parameters()
        self.ase_calculator = Calculator.from_checkpoint(self.model_path)
