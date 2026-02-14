try:
    from .bec import LoremBEC
    from .mlip import Lorem
except ModuleNotFoundError:
    pass

__all__ = ["Lorem", "LoremBEC"]
