import unittest

import jax
import numpy as np
from omegaconf import OmegaConf

from lagrangebench.train.strats import push_forward_sample_steps


class TestPushForward(unittest.TestCase):
    """Class for unit testing the push-forward functions."""

    def setUp(self):
        self.pf = OmegaConf.create(
            {
                "steps": [-1, 20000, 50000, 100000],
                "unrolls": [0, 1, 3, 20],
                "probs": [4.05, 4.05, 1.0, 1.0],
            }
        )

        self.key = jax.random.PRNGKey(42)

    def body_steps(self, step, unrolls, probs):
        dump = []
        for _ in range(1000):
            self.key, unroll_steps = push_forward_sample_steps(self.key, step, self.pf)
            dump.append(unroll_steps)

        # Note: np.unique returns sorted array
        unique, counts = np.unique(dump, return_counts=True)
        self.assertTrue((unique == unrolls).all(), "Wrong unroll steps")
        self.assertTrue(
            np.allclose(counts / 1000, probs, atol=0.05),
            "Wrong probabilities of unroll steps",
        )

    def test_pf_step_1(self):
        self.body_steps(1, np.array([0]), np.array([1.0]))

    def test_pf_step_60000(self):
        self.body_steps(60000, np.array([0, 1, 3]), np.array([0.45, 0.45, 0.1]))


if __name__ == "__main__":
    unittest.main()
