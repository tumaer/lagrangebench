import unittest

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import numpy as np

from lagrangebench import models
from lagrangebench.utils import NodeType


class ModelTest(unittest.TestCase):
    def dummy_sample(self, vel=None, pos=None):
        key = self.key()

        if vel is None:
            vel = jax.random.uniform(key, (100, 5 * 3))
        if pos is None:
            pos = jax.random.uniform(key, (100, 1, 3))

        senders = jax.random.randint(key, (200,), 0, 100)
        receivers = jax.random.randint(key, (200,), 0, 100)
        rel_disp = (pos[receivers] - pos[senders]).squeeze()
        x = {
            "vel_hist": vel,
            "vel_mag": jnp.sum(vel.reshape(100, -1, 3) ** 2, -1) ** 0.5,
            "rel_disp": rel_disp,
            "rel_dist": jnp.sum(rel_disp**2, -1, keepdims=True) ** 0.5,
            "abs_pos": pos,
            "senders": senders,
            "receivers": receivers,
        }
        particle_type = jnp.ones((100, 1), dtype=jnp.int32) * NodeType.FLUID
        return x, particle_type

    def key(self):
        return jax.random.PRNGKey(0)

    def assert_equivariant(self, model):
        key = self.key()

        vel = e3nn.normal("5x1o", key, (100,))
        pos = e3nn.normal("1x1o", key, (100,))

        def wrapper(v, p):
            sample, particle_type = self.dummy_sample()
            sample.update(
                {
                    "vel_hist": v.array.reshape((100, 5 * 3)),
                    "abs_pos": p.array.reshape((100, 1, 3)),
                }
            )
            y = model((sample, particle_type))
            return e3nn.IrrepsArray("1x1o", y["acc"])

        # random rotation matrix
        R = -e3nn.rand_matrix(key, ())

        out1 = wrapper(vel.transform_by_matrix(R), pos.transform_by_matrix(R))
        out2 = wrapper(vel, pos).transform_by_matrix(R)

        def assert_(x, y):
            self.assertTrue(
                np.isclose(x, y, atol=1e-5, rtol=1e-5).all(), "Not equivariant!"
            )

        jax.tree_util.tree_map(assert_, out1, out2)

    def test_segnn(self):
        segnn = models.SEGNN(
            node_features_irreps="5x1o + 5x0e",
            edge_features_irreps="1x1o + 1x0e",
            scalar_units=8,
            lmax_hidden=1,
            lmax_attributes=1,
            n_vels=5,
            num_mp_steps=1,
            output_irreps="1x1o",
            key=self.key(),
        )
        self.assert_equivariant(segnn)

    def test_egnn(self):
        egnn = models.EGNN(
            hidden_size=8,
            output_size=1,
            num_mp_steps=1,
            dt=0.01,
            n_vels=5,
            displacement_fn=lambda x, y: x - y,
            shift_fn=lambda x, y: x + y,
            key=self.key(),
        )
        self.assert_equivariant(egnn)

    def test_painn(self):
        # With homogeneous particles: s_in = n_vels (5); no force/bound → v_in = 5
        painn = models.PaiNN(
            hidden_size=8,
            output_size=1,
            num_mp_steps=1,
            radial_basis_fn=models.painn.gaussian_rbf(20, 10, trainable=True),
            cutoff_fn=models.painn.cosine_cutoff(10),
            n_vels=5,
            n_rbf=20,
            s_in_size=5,
            v_in_channels=5,
            key=self.key(),
        )
        self.assert_equivariant(painn)


if __name__ == "__main__":
    unittest.main()
