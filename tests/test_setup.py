import json
import os
from argparse import Namespace

import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap
from jax_md import space
from torch.utils.data import DataLoader

from gns_jax.data import H5Dataset, numpy_collate
from gns_jax.utils import broadcast_from_batch, eval_single_rollout, setup_builder


class TestSetupBuilder:
    """Class for unit testing the setup functions."""

    def __init__(self):

        args = Namespace()
        args.normalization = {
            "acceleration": {
                "mean": np.zeros((3,)),
                "std": np.ones((3,)),
            },
            "velocity": {
                "mean": np.zeros((3,)),
                "std": np.ones((3,)),
            },
        }
        args.metadata = {
            "periodic_boundary_conditions": [True, True, True],
            "default_connectivity_radius": 0.3,
            "bounds": [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        }
        args.config = Namespace()
        args.config.magnitudes = False
        args.config.log_norm = False
        args.config.input_seq_length = 3  # one past velocity
        bounds = np.array(args.metadata["bounds"])
        args.box = bounds[:, 1] - bounds[:, 0]

        self.args = args
        external_force_fn = None
        self.setup = setup_builder(args, external_force_fn)
        self.key = jax.random.PRNGKey(0)

        # position input shape (num_particles, sequence_len, dim) = (3, 5, 3)
        self.position_data = np.array(
            [
                [
                    [0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5],
                ],
                [
                    [0.7, 0.5, 0.5],
                    [0.9, 0.5, 0.5],
                    [0.1, 0.5, 0.5],
                    [0.3, 0.5, 0.5],
                    [0.5, 0.5, 0.5],
                ],
                [
                    [0.8, 0.6, 0.5],
                    [0.8, 0.6, 0.5],
                    [0.9, 0.6, 0.5],
                    [0.2, 0.6, 0.5],
                    [0.6, 0.6, 0.5],
                ],
            ]
        )
        self.particle_types = np.array([0, 0, 0])

    def test_allocate(self):

        # test PBC and velocity and acceleration computation without noise
        key, features, target_dict, neighbors = self.setup.allocate(
            self.key, (self.position_data, self.particle_types)
        )
        assert (
            neighbors.idx == jnp.array([[0, 1, 2, 2, 1, 3], [0, 1, 1, 2, 2, 3]])
        ).all()

        assert (key != self.key).all()

        assert jnp.isclose(
            target_dict["vel"],
            jnp.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]),
        ).all()

        assert jnp.isclose(
            target_dict["acc"],
            jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]),
        ).all()

        assert jnp.isclose(
            features["vel_hist"],
            jnp.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.2, 0.0, 0.0, 0.2, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.1, 0.0, 0.0],
                ]
            ),
        ).all()

        most_recent_displacement = jnp.array(
            [
                [0.0, 0.0, 0.0],  # edge 0-0
                [0.0, 0.0, 0.0],  # edge 1-1
                [-0.2, 0.1, 0.0],  # edge 2-1
                [0.0, 0.0, 0.0],  # edge 2-2
                [0.2, -0.1, 0.0],  # edge 1-2
                [0.0, 0.0, 0.0],  # edge 3-3
            ]
        )
        r0 = self.args.metadata["default_connectivity_radius"]
        normalized_displ = most_recent_displacement / r0
        normalized_dist = ((normalized_displ**2).sum(-1, keepdims=True)) ** 0.5

        assert jnp.isclose(features["rel_disp"], normalized_displ).all()
        assert jnp.isclose(features["rel_dist"], normalized_dist).all()

        print("test_allocate passed!")

        self.neighbors = neighbors

    def test_preprocess_base(self):
        # preprocess is 1-to-1 the same as allocate, up to the neighbors' computation
        _, _, _, neighbors_new = self.setup.preprocess(
            self.key, (self.position_data, self.particle_types), 0.0, self.neighbors, 0
        )

        assert (self.neighbors.idx == neighbors_new.idx).all()

        print("test_preprocess_base passed!")

    def test_preprocess_unroll(self):
        # test getting the second available target acceleration
        _, _, target_dict, _ = self.setup.preprocess(
            self.key, (self.position_data, self.particle_types), 0.0, self.neighbors, 1
        )

        assert jnp.isclose(
            target_dict["acc"],
            jnp.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.1, 0.0, 0.0],
                ]
            ),
            atol=1e-07,
        ).all()

        print("test_preprocess_unroll passed!")

    def test_preprocess_noise(self):
        # test that both potential targets are corrected with the proper noise
        # we choose noise_std=0.01 to guarantee that no particle with jump periodically
        _, features, target_dict, _ = self.setup.preprocess(
            self.key, (self.position_data, self.particle_types), 0.01, self.neighbors, 0
        )
        vel_next1 = jnp.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]])
        correct_target_acc = vel_next1 - features["vel_hist"][:, 3:6]
        assert jnp.isclose(correct_target_acc, target_dict["acc"], atol=1e-7).all()

        # with one push-forward step on top
        _, features, target_dict, _ = self.setup.preprocess(
            self.key, (self.position_data, self.particle_types), 0.01, self.neighbors, 1
        )
        vel_next2 = jnp.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0]])
        correct_target_acc = vel_next2 - vel_next1
        assert jnp.isclose(correct_target_acc, target_dict["acc"], atol=1e-7).all()

        print("test_preprocess_noise passed!")

    def test_allocate_eval(self):
        pass

    def test_preprocess_eval(self):
        pass

    def test_integrate(self):
        # given the reference acceleration, compute the next position
        correct_acceletation = jnp.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]
        )

        new_pos = self.setup.integrate(correct_acceletation, self.position_data[:, :3])

        assert jnp.isclose(new_pos, self.position_data[:, 3]).all()

        print("test_integrate passed!")

    def run(self):
        self.test_allocate()
        self.test_preprocess_base()
        self.test_preprocess_unroll()
        self.test_preprocess_noise()
        self.test_allocate_eval()
        self.test_preprocess_eval()
        self.test_integrate()


class TestInferBuilder:
    """Class for unit testing the evaluate_single_rollout function."""

    def __init__(self):

        # the LJ debug dataset containes 3 particles over 405 time instances
        data_dir = "/home/atoshev/data/LJ_debug"

        args = Namespace()
        args.normalization = {
            "acceleration": {
                "mean": np.zeros((3,)),
                "std": np.ones((3,)),
            },
            "velocity": {
                "mean": np.zeros((3,)),
                "std": np.ones((3,)),
            },
        }
        with open(os.path.join(data_dir, "metadata.json"), "r") as f:
            args.metadata = json.loads(f.read())

        args.config = Namespace()
        args.config.data_dir = data_dir
        args.config.magnitudes = False
        args.config.log_norm = False
        args.config.input_seq_length = 3  # one past velocity
        args.config.metrics = ["mse"]
        args.config.num_rollout_steps = (
            args.metadata["sequence_length"] - args.config.input_seq_length
        )

        bounds = np.array(args.metadata["bounds"])
        args.box = bounds[:, 1] - bounds[:, 0]

        self.args = args
        external_force_fn = None
        self.setup = setup_builder(args, external_force_fn)
        self.key = jax.random.PRNGKey(0)

        self.displacement_fn, self.shift_fn = space.periodic(side=args.box)

        data_valid = H5Dataset(
            args.config.data_dir, "valid", args.config.input_seq_length, is_rollout=True
        )
        self.loader_valid = DataLoader(
            dataset=data_valid, batch_size=1, collate_fn=numpy_collate
        )

    def test_rollout(self):
        isl = self.loader_valid.dataset.input_seq_length
        num_rollout_steps = self.args.config.num_rollout_steps + 1

        # get one validation trajectory from the debug dataset
        traj_i = next(iter(self.loader_valid))
        # remove batch dimension
        assert traj_i[0].shape[0] == 1, "Batch dimension must be 1"
        traj_i = broadcast_from_batch(traj_i, index=0)

        positions = traj_i[0]  # (nodes, t, dim) = (3, 405, 3)

        displ_vmap = vmap(self.displacement_fn, (0, 0))
        displ_dvmap = vmap(displ_vmap, (0, 0))

        vels = displ_dvmap(positions[:, 1:], positions[:, :-1])  # (3, 404, 3)
        accs = vels[:, 1:] - vels[:, :-1]  # (3, 403, 3)

        def UnitTestModel(accelecations, idx_start, idx_len):
            acc_iterator = iter(np.arange(idx_start, idx_start + idx_len))

            def model_apply(_, b, __):
                i = next(acc_iterator)
                acc_i = accelecations[:, i]
                return acc_i, b

            return model_apply

        #######################################################################
        # proof that the above "model" works
        model_apply = UnitTestModel(accs, isl - 2, num_rollout_steps)
        pred_acc = model_apply(None, None, None)[0]
        pred_pos = self.shift_fn(positions[:, isl - 1], vels[:, isl - 2] + pred_acc)
        pred_pos = jnp.asarray(pred_pos, dtype=jnp.float32)
        target_pos = positions[:, isl]

        assert jnp.isclose(pred_pos, target_pos, atol=1e-7).all()
        #######################################################################

        model_apply = UnitTestModel(accs, isl - 2, num_rollout_steps)
        _, neighbors = self.setup.allocate_eval((positions[:, :isl], traj_i[1]))

        example_rollout, metrics, neighbors = eval_single_rollout(
            setup=self.setup,
            model_apply=model_apply,
            params=None,
            state=None,
            neighbors=neighbors,
            traj_i=traj_i,
            num_rollout_steps=num_rollout_steps,
            input_seq_length=isl,
            graph_preprocess=lambda x, y: 0,
            eval_n_more_steps=0,
            oversmooth_norm_hops=0,
        )
        print(f"Average rollout mse: {metrics['mse'].mean()}")

        pos_input = traj_i[0].transpose(1, 0, 2)  # (t, nodes, dim)
        initial_positions = pos_input[: self.loader_valid.dataset.input_seq_length]
        example_full = np.concatenate([initial_positions, example_rollout], axis=0)
        rollout_dict = {
            "predicted_rollout": example_full,  # (t, nodes, dim)
            "ground_truth_rollout": pos_input,  # (t, nodes, dim)
        }

        assert jnp.isclose(
            rollout_dict["predicted_rollout"][100],
            rollout_dict["ground_truth_rollout"][100],
            atol=1e-7,
        ).all()

        print("test_rollout passed!")

    def run(self):
        self.test_rollout()


if __name__ == "__main__":
    TestSetupBuilder().run()
    TestInferBuilder().run()
