from typing import Dict, Tuple

import haiku as hk
import jax.numpy as jnp
import numpy as np
from jax import vmap


class Linear(hk.Module):
    """Model defining linear relation between input nodes and targets."""

    def __init__(self, dim_out):
        super().__init__()
        self.mlp = hk.nets.MLP([dim_out], activate_final=False, name="MLP")

    def __call__(
        self, sample: Tuple[Dict[str, jnp.ndarray], np.ndarray]
    ) -> jnp.ndarray:
        # transform
        features, particle_type = sample
        x = [
            features[k]
            for k in ["vel_hist", "vel_mag", "bound", "force"]
            if k in features[0]
        ] + [particle_type]
        # call
        return vmap(self.mlp)(jnp.concatenate(x, axis=-1))
