"""NORA: node-wise optimization, pairwise reconstruction, and coordination."""

from dataclasses import dataclass

from certfix3_rc6.config import Config as _Config

__version__ = "0.1.0"
__all__ = ["Config", "fit", "__version__"]


@dataclass(frozen=True)
class Config(_Config):
    """Frozen solver settings with the six-hour budget used in the paper."""

    hot_cap_seconds: float = 21600.0


def fit(X, *, config=None):
    """Fit NORA to an (n, d) array; return weights, graph, and diagnostics.

    W[i, j] denotes i -> j. W is unthresholded; graph is the binary output.
    Check result['numerically_qualified'] before using a completed DAG result.
    """
    from certfix3_rc6 import fit as _fit

    return _fit(X, config=Config() if config is None else config)
