"""Base model class. All models should inherit from this class."""

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import haiku as hk
import jax.numpy as jnp


class BaseModel(hk.Module, ABC):
    """Base model class. All models must inherit from this class."""

    @abstractmethod
    def __call__(
        self, sample: Tuple[Dict[str, jnp.ndarray], jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        """Forward pass.

        Args:
            sample: Tuple with feature dictionary and particle type. Possible features:
                * "abs_pos", bsolute positions
                * "vel_hist", historical velocity sequence
                * "vel_mag", velocity magnitudes
                * "bound", distance to boundaries
                * "force", external force field
                * "rel_disp", relative displacement vectors
                * "rel_dist", relative distance vectors
        Returns:
            Dict with model output. The keys must be at least one of:
                * "acc", (normalized) acceleration
                * "vel", (normalized) velocity
                * "pos", (absolute) next position
        """
        raise NotImplementedError
