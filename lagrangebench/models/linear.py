"""Simple baseline linear model (Equinox)."""

from typing import Dict, Tuple

import jax.numpy as jnp
import numpy as np

from .base import BaseModel
from .utils import LinearXav


class Linear(BaseModel):
    r"""Linear relation between input scalar features and targets.

    :math:`\mathbf{a}_i = \mathbf{W} \mathbf{x}_i`.
    """

    mlp: LinearXav

    def __init__(self, in_features: int, dim_out: int, *, key):
        """Initialize the model.

        Args:
            in_features: Total number of scalar input features per node (sum of the
                dimensionalities of ``vel_hist``, ``vel_mag``, ``bound``, ``force``
                plus 1 for the particle type).
            dim_out: Output dimensionality.
            key: PRNG key for init.
        """
        self.mlp = LinearXav(in_features, dim_out, key=key)

    def __call__(
        self, sample: Tuple[Dict[str, jnp.ndarray], np.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        features, particle_type = sample
        x = [
            features[k]
            for k in ["vel_hist", "vel_mag", "bound", "force"]
            if k in features
        ] + [particle_type[:, None]]
        acc = self.mlp(jnp.concatenate(x, axis=-1))
        return {"acc": acc}
