"""One Column--Reconstruction--Column pass for linear DAG learning."""
__version__ = "crc-p-block-full-threshold005-numfix2"
from .config import Config

__all__ = ["Config", "fit", "refine", "__version__"]

def fit(X, **kwargs):
    from .api import fit as run
    return run(X, **kwargs)

def refine(X, W, **kwargs):
    from .api import refine as run
    return run(X, W, **kwargs)
