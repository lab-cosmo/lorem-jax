from opsis.comms import Comms

from ._version import __version__
from .batching import to_batch, to_sample
from .calculator import Calculator as LOREMCalculator
from .models.bec import LoremBEC
from .models.mlip import Lorem
from .transforms import ToBatch, ToSample

comms = Comms("lorem")

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
