"""LOREM driver for i-PI."""

import numpy as np

import json

from ipi.pes.ase import ASEDriver

from lorem.calculator import Calculator

__DRIVER_NAME__ = "lorem"
__DRIVER_CLASS__ = "LOREM_driver"


class LOREM_driver(ASEDriver):
    def __init__(self, template, model_path, *args, skin=0.25, **kwargs):
        self.model_path = model_path
        self.skin = skin
        super().__init__(template, *args, **kwargs)
        self.capabilities.append("born_effective_charges")

    def check_parameters(self):
        super().check_parameters()
        has_stress = "stress" in self.capabilities
        self.ase_calculator = Calculator.from_checkpoint(
            self.model_path, stress=has_stress, skin=self.skin
        )

    def compute_structure(self, cell, pos):
        pot_ipi, force_ipi, vir_ipi, extras = super().compute_structure(cell, pos)

        # BEC passthrough: when model outputs "born_effective_charges" (e.g. LoremBEC),
        # expose as "BEC" in (3*natoms, 3) layout for i-PI compatibility
        if extras and "born_effective_charges" in extras:
            extras = json.loads(extras)
            BEC = np.reshape(extras.pop("born_effective_charges"), (-1, 3))
            extras["BEC"] = BEC.tolist()
            extras = json.dumps(extras)

        return pot_ipi, force_ipi, vir_ipi, extras
