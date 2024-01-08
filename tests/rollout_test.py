import unittest
from argparse import Namespace

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax import config as jax_config
from jax import jit, vmap
from jax_md import space
from torch.utils.data import DataLoader

jax_config.update("jax_enable_x64", True)

from lagrangebench.case_setup import case_builder
from lagrangebench.data import H5Dataset
from lagrangebench.data.utils import get_dataset_stats, numpy_collate
from lagrangebench.evaluate import MetricsComputer
from lagrangebench.evaluate.rollout import eval_batched_rollout
from lagrangebench.utils import broadcast_from_batch


class TestInferBuilder(unittest.TestCase):
    """Class for unit testing the evaluate_single_rollout function."""

    def setUp(self):
        self.config = Namespace(
            data_dir="tests/3D_LJ_3_1214every1",  # Lennard-Jones dataset
            input_seq_length=3,  # two past velocities
            metrics=["mse"],
            n_rollout_steps=100,
            isotropic_norm=False,
            noise_std=0.0,
        )

        data_valid = H5Dataset(
            split="valid",
            dataset_path=self.config.data_dir,
            name="lj3d",
            input_seq_length=self.config.input_seq_length,
            extra_seq_length=self.config.n_rollout_steps,
        )
        self.loader_valid = DataLoader(
            dataset=data_valid, batch_size=1, collate_fn=numpy_collate
        )

        self.metadata = data_valid.metadata
        self.normalization_stats = get_dataset_stats(
            self.metadata, self.config.isotropic_norm, self.config.noise_std
        )

        bounds = np.array(self.metadata["bounds"])
        box = bounds[:, 1] - bounds[:, 0]
        self.displacement_fn, self.shift_fn = space.periodic(side=box)

        self.case = case_builder(
            box,
            self.metadata,
            self.config.input_seq_length,
            noise_std=self.config.noise_std,
        )

        self.key = jax.random.PRNGKey(0)

    def test_rollout(self):
        isl = self.loader_valid.dataset.input_seq_length

        # get one validation trajectory from the debug dataset
        traj_batch_i = next(iter(self.loader_valid))
        traj_batch_i = jax.tree_map(lambda x: jnp.array(x), traj_batch_i)
        # remove batch dimension
        self.assertTrue(traj_batch_i[0].shape[0] == 1, "We test only batch size 1")
        traj_i = broadcast_from_batch(traj_batch_i, index=0)
        positions = traj_i[0]  # (nodes, t, dim) = (3, 405, 3)

        displ_vmap = vmap(self.displacement_fn, (0, 0))
        displ_dvmap = vmap(displ_vmap, (0, 0))
        vels = displ_dvmap(positions[:, 1:], positions[:, :-1])  # (3, 404, 3)
        accs = vels[:, 1:] - vels[:, :-1]  # (3, 403, 3)
        stats = self.normalization_stats["acceleration"]
        accs = (accs - stats["mean"]) / stats["std"]

        class CheatingModel(hk.Module):
            def __init__(self, target, start):
                super().__init__()
                self.target = target
                self.start = start

            def __call__(self, x):
                i = hk.get_state(
                    "counter",
                    shape=[],
                    dtype=jnp.int32,
                    init=hk.initializers.Constant(self.start),
                )
                hk.set_state("counter", i + 1)
                return {"acc": self.target[:, i]}

        def setup_model(target, start):
            def model(x):
                return CheatingModel(target, start)(x)

            model = hk.without_apply_rng(hk.transform_with_state(model))
            params, state = model.init(None, None)
            model_apply = model.apply
            model_apply = jit(model_apply)
            return params, state, model_apply

        params, state, model_apply = setup_model(accs, 0)

        # proof that the above "model" works
        out, state = model_apply(params, state, None)
        pred_acc = stats["mean"] + out["acc"] * stats["std"]
        pred_pos = self.shift_fn(positions[:, isl - 1], vels[:, isl - 2] + pred_acc)
        pred_pos = jnp.asarray(pred_pos, dtype=jnp.float32)
        target_pos = positions[:, isl]

        assert jnp.isclose(pred_pos, target_pos, atol=1e-7).all(), "Wrong setup"

        params, state, model_apply = setup_model(accs, isl - 2)
        _, neighbors = self.case.allocate_eval((positions[:, :isl], traj_i[1]))

        metrics_computer = MetricsComputer(
            ["mse"],
            self.case.displacement,
            self.metadata,
            isl,
        )

        example_rollout_batch, metrics_batch, neighbors = eval_batched_rollout(
            model_apply=model_apply,
            case=self.case,
            params=params,
            state=state,
            traj_batch_i=traj_batch_i,
            neighbors=neighbors,
            metrics_computer=metrics_computer,
            n_rollout_steps=self.config.n_rollout_steps,
            t_window=isl,
        )
        example_rollout = broadcast_from_batch(example_rollout_batch, index=0)
        metrics = broadcast_from_batch(metrics_batch, index=0)

        self.assertTrue(
            jnp.isclose(
                metrics["mse"].mean(),
                jnp.array(0.0),
                atol=1e-6,
            ).all(),
            "Wrong rollout mse",
        )

        pos_input = traj_i[0].transpose(1, 0, 2)  # (t, nodes, dim)
        initial_positions = pos_input[:isl]
        example_full = np.concatenate([initial_positions, example_rollout], axis=0)
        rollout_dict = {
            "predicted_rollout": example_full,  # (t, nodes, dim)
            "ground_truth_rollout": pos_input,  # (t, nodes, dim)
        }

        self.assertTrue(
            jnp.isclose(
                rollout_dict["predicted_rollout"][100, 0],
                rollout_dict["ground_truth_rollout"][100, 0],
                atol=1e-6,
            ).all(),
            "Wrong rollout prediction",
        )


if __name__ == "__main__":
    unittest.main()
