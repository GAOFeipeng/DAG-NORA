"""Exercise the public entry point through a complete fit."""

import unittest

import numpy as np

from nora import Config, fit


class SmokeTest(unittest.TestCase):
    def test_complete_fit(self):
        rng = np.random.default_rng(617)
        X = rng.normal(size=(64, 5))
        X[:, 1] += 0.9 * X[:, 0]
        X[:, 3] -= 0.7 * X[:, 1]
        original = X.copy()

        config = Config()
        result = fit(X, config=config)

        self.assertEqual(config.hot_cap_seconds, 21600)
        self.assertTrue(result["final_returned"], result.get("exception"))
        self.assertTrue(result["numerically_qualified"], result["status"])
        self.assertTrue(result["dag"])
        self.assertEqual(len(result["stages"]), 5)
        self.assertTrue(result["joint_executed"])
        self.assertEqual(result["W"].shape, (5, 5))
        self.assertTrue(np.isfinite(result["W"]).all())
        np.testing.assert_array_equal(X, original)
        np.testing.assert_array_equal(result["graph"], np.abs(result["W"]) >= config.threshold)
        np.testing.assert_array_equal(np.diag(result["graph"]), np.zeros(5))


if __name__ == "__main__":
    unittest.main()
