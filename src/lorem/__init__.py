from ._version import __version__

try:
    from comms import Comms

    comms = Comms("lorem")
except ModuleNotFoundError:
    comms = None

from .dynamics import LOREMCalculator

# These modules require marathon and jaxpme at import time.
# When those are not installed (e.g. CI), we skip them.
try:
    from .batching import to_batch, to_sample
    from .models.bec import LoremBEC
    from .models.mlip import Lorem
    from .transforms import ToBatch, ToSample
except ModuleNotFoundError:
    pass

__all__ = [
    "__version__",
    "LOREMCalculator",
    "Lorem",
    "LoremBEC",
    "to_sample",
    "to_batch",
    "ToSample",
    "ToBatch",
]
