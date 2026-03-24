from opsis.comms import Comms

from ._version import __version__
from .batching import to_batch, to_sample
from .models.bec import LoremBEC
from .models.mlip import Lorem
from .transforms import ToBatch, ToSample

comms = Comms("lorem")


def __getattr__(name):
    if name == "LOREMCalculator":
        from .calculator import Calculator

        return Calculator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "__version__",
    "Lorem",
    "LoremBEC",
    "LOREMCalculator",
    "to_sample",
    "to_batch",
    "ToSample",
    "ToBatch",
]
