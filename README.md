# NORA

Reference implementation of *Learning Directed Acyclic Graphs via Exact
Node-wise Reduction and Pairwise Reconstruction*.

NORA learns a linear DAG through cyclic node optimization, pairwise
reconstruction, and final node coordination.

## Installation

Requires Python 3.11 or newer, Linux x86-64 (including WSL2), and GCC with
`libquadmath`. On Ubuntu, the system prerequisites are provided by
`build-essential` and `python3-venv`.

```bash
git clone https://github.com/GAOFeipeng/DAG-NORA.git
cd DAG-NORA
python3 -m venv .venv
source .venv/bin/activate
python tools/build.py
python -m pip install -e .
python examples/minimal.py
```

The first fit includes Numba compilation. Run fits on the main thread;
use separate processes for parallel fits. Numerical libraries are limited
to one thread during fitting.

## Usage

```python
import numpy as np
from nora import Config, fit

X = np.load("X.npy")  # shape: (samples, variables)
result = fit(X, config=Config())

if not result["numerically_qualified"]:
    raise RuntimeError(f"Fit did not complete with a valid DAG: {result['status']}")

W = result["W"]          # unthresholded continuous weights
graph = result["graph"]  # binary adjacency matrix
```

`W[i, j]` represents the edge `i -> j`. Input must be a finite numeric matrix
with at least two samples and two variables. Columns are centered internally;
their variances are not standardized. The input array is not modified.

Defaults match the paper: `lambda1=0.03`, output threshold `0.3`, five
continuation stages with initial `s=(1, 0.9, 0.8, 0.7, 0.6)` and
`mu=(1, 0.1, 0.01, 0.001, 0.0001)`, and a six-hour fitting budget, excluding
initial compilation. Reconstruction uses pair threshold `0.05` and the top eight
candidates. Settings can be supplied through
`Config`, for example `Config(lambda1=0.03, hot_cap_seconds=21600)`.

The graph uses `abs(W) >= threshold`. Check `numerically_qualified` before
using it as a completed DAG result. `status`, `termination`, and
`residual_check` provide diagnostics; incomplete runs can provide a separately
labelled `partial_result`.

Run the smoke test with:

```bash
python -m unittest discover -s tests -v
```

## Citation

```bibtex
@unpublished{gao2026nora,
  title = {Learning Directed Acyclic Graphs via Exact Node-wise Reduction and Pairwise Reconstruction},
  author = {Gao, Feipeng and Cui, Yani and Wang, Xiaokang and Ren, Jia},
  year = {2026},
  note = {Manuscript}
}
```

## License

Apache-2.0. The implementation includes an adapted DAGMA continuation
controller; see [NOTICE](NOTICE) for attribution.
