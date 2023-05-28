from argparse import Namespace
from typing import Dict, Tuple, Type

import haiku as hk
import jax.numpy as jnp
import numpy as np
from jax import vmap

from .base import BaseModel


class Linear(BaseModel):
    """Model defining linear relation between input nodes and targets."""

    def __init__(self, dim_out):
        super().__init__()
        self.mlp = hk.Linear(dim_out)

    def __call__(
        self, sample: Tuple[Dict[str, jnp.ndarray], np.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        # transform
        features, particle_type = sample
        x = [
            features[k]
            for k in ["vel_hist", "vel_mag", "bound", "force"]
            if k in features
        ] + [particle_type[:, None]]
        # call
        acc = vmap(self.mlp)(jnp.concatenate(x, axis=-1))
        return {"acc": acc}

    @classmethod
    def setup_model(cls, args: Namespace) -> Tuple["Linear", Type]:
        _ = args
        return cls(dim_out=args.metadata["dim"])
