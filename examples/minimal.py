"""Generate a small linear dataset and fit NORA with the paper defaults."""

import numpy as np

from nora import fit


def main():
    rng = np.random.default_rng(617)
    X = rng.normal(size=(1000, 5))
    X[:, 1] += 0.9 * X[:, 0]
    X[:, 3] -= 0.7 * X[:, 1]

    result = fit(X)
    if not result["numerically_qualified"]:
        raise RuntimeError(
            f"Fit did not complete with a valid DAG: {result['status']}; "
            f"{result.get('exception', result['termination'])}"
        )
    print("Status:", result["status"])
    print("Edges:", int(result["graph"].sum()))
    print("Adjacency matrix (row -> column):")
    print(result["graph"])


if __name__ == "__main__":
    main()
