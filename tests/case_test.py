import unittest

import jax
import jax.numpy as jnp
import numpy as np

from lagrangebench.case_setup import case_builder


class TestCaseBuilder(unittest.TestCase):
    """Class for unit testing the case builder functions."""

    def setUp(self):
        self.metadata = {
            "num_particles_max": 3,
            "periodic_boundary_conditions": [True, True, True],
            "default_connectivity_radius": 0.3,
            "bounds": [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            "acc_mean": [0.0, 0.0, 0.0],
            "acc_std": [1.0, 1.0, 1.0],
            "vel_mean": [0.0, 0.0, 0.0],
            "vel_std": [1.0, 1.0, 1.0],
        }

        bounds = np.array(self.metadata["bounds"])
        box = bounds[:, 1] - bounds[:, 0]

        self.case = case_builder(
            box,
            self.metadata,
            input_seq_length=3,  # two past velocities
            cfg_neighbors={"backend": "jaxmd_vmap", "multiplier": 1.25},
            cfg_model={"isotropic_norm": False, "magnitude_features": False},
            noise_std=0.0,
            external_force_fn=None,
        )
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

        _, _, _, neighbors = self.case.allocate(
            self.key, (self.position_data, self.particle_types)
        )
        self.neighbors = neighbors

    def test_allocate(self):
        # test PBC and velocity and acceleration computation without noise
        key, features, target_dict, neighbors = self.case.allocate(
            self.key, (self.position_data, self.particle_types)
        )
        self.assertTrue(
            (
                neighbors.idx == jnp.array([[0, 1, 2, 2, 1, 3], [0, 1, 1, 2, 2, 3]])
            ).all(),
            "Wrong edge list after allocate",
        )

        self.assertTrue((key != self.key).all(), "Key not updated at allocate")

        self.assertTrue(
            jnp.isclose(
                target_dict["vel"],
                jnp.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]),
            ).all(),
            "Wrong target velocity at allocate",
        )

        self.assertTrue(
            jnp.isclose(
                target_dict["acc"],
                jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]),
            ).all(),
            "Wrong target acceleration at allocate",
        )

        self.assertTrue(
            jnp.isclose(
                features["vel_hist"],
                jnp.array(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # particle 1, two past vels.
                        [0.2, 0.0, 0.0, 0.2, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.1, 0.0, 0.0],
                    ]
                ),
            ).all(),
            "Wrong historic velocities at allocate",
        )

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
        r0 = self.metadata["default_connectivity_radius"]
        normalized_displ = most_recent_displacement / r0
        normalized_dist = ((normalized_displ**2).sum(-1, keepdims=True)) ** 0.5

        self.assertTrue(
            jnp.isclose(features["rel_disp"], normalized_displ).all(),
            "Wrong relative displacement at allocate",
        )
        self.assertTrue(
            jnp.isclose(features["rel_dist"], normalized_dist).all(),
            "Wrong relative distance at allocate",
        )

    def test_preprocess_base(self):
        # preprocess is 1-to-1 the same as allocate, up to the neighbors' computation
        _, _, _, neighbors_new = self.case.preprocess(
            self.key, (self.position_data, self.particle_types), 0.0, self.neighbors, 0
        )

        self.assertTrue(
            (self.neighbors.idx == neighbors_new.idx).all(),
            "Wrong edge list after preprocess",
        )

    def test_preprocess_unroll(self):
        # test getting the second available target acceleration
        _, _, target_dict, _ = self.case.preprocess(
            self.key, (self.position_data, self.particle_types), 0.0, self.neighbors, 1
        )

        self.assertTrue(
            jnp.isclose(
                target_dict["acc"],
                jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
                atol=1e-07,
            ).all(),
            "Wrong target acceleration at preprocess",
        )

    def test_preprocess_noise(self):
        # test that both potential targets are corrected with the proper noise
        # we choose noise_std=0.01 to guarantee that no particle will jump periodically
        _, features, target_dict, _ = self.case.preprocess(
            self.key, (self.position_data, self.particle_types), 0.01, self.neighbors, 0
        )
        vel_next1 = jnp.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]])
        correct_target_acc = vel_next1 - features["vel_hist"][:, 3:6]
        self.assertTrue(
            jnp.isclose(correct_target_acc, target_dict["acc"], atol=1e-7).all(),
            "Wrong target acceleration at preprocess",
        )

        # with one push-forward step on top
        _, features, target_dict, _ = self.case.preprocess(
            self.key, (self.position_data, self.particle_types), 0.01, self.neighbors, 1
        )
        vel_next2 = jnp.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0]])
        correct_target_acc = vel_next2 - vel_next1
        self.assertTrue(
            jnp.isclose(correct_target_acc, target_dict["acc"], atol=1e-7).all(),
            "Wrong target acceleration at preprocess with 1 pushforward step",
        )

    def test_allocate_eval(self):
        pass

    def test_preprocess_eval(self):
        pass

    def test_integrate(self):
        # given the reference acceleration, compute the next position
        correct_acceletation = {
            "acc": jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
        }

        new_pos = self.case.integrate(correct_acceletation, self.position_data[:, :3])

        self.assertTrue(
            jnp.isclose(new_pos, self.position_data[:, 3]).all(),
            "Wrong new position at integration",
        )


if __name__ == "__main__":
    unittest.main()
