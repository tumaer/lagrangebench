"""Runner test with a linear model and LJ dataset."""

import unittest

from omegaconf import OmegaConf

from lagrangebench.defaults import defaults
from lagrangebench.runner import train_or_infer


class TestRunner(unittest.TestCase):
    """Test whether train_or_infer runs through."""

    def setUp(self):
        self.cfg = OmegaConf.create(
            {
                "mode": "all",
                "dataset": {
                    "src": "tests/3D_LJ_3_1214every1",
                },
                "model": {
                    "name": "linear",
                    "input_seq_length": 3,
                },
                "train": {
                    "step_max": 10,
                    "noise_std": 0.0,
                },
                "eval": {
                    "n_rollout_steps": 5,
                    "train": {
                        "n_trajs": 2,
                        "metrics_stride": 5,
                        "metrics": ["mse"],
                        "out_type": "none",
                    },
                    "infer": {
                        "n_trajs": 2,
                        "metrics_stride": 1,
                        "metrics": ["mse"],
                        "out_type": "none",
                    },
                },
                "logging": {
                    "log_steps": 1,
                    "eval_steps": 5,
                    "wandb": False,
                    "ckp_dir": "/tmp/ckp",
                },
            }
        )
        # overwrite defaults with user-defined config
        self.cfg = OmegaConf.merge(defaults, self.cfg)

    def test_runner(self):
        out = train_or_infer(self.cfg)
        self.assertEqual(out, 0)


if __name__ == "__main__":
    unittest.main()
