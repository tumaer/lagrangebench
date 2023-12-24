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

        We specify the dimensions of the inputs and outputs using the number of nodes N,
        the number of edges E, number of historic velocities K (=input_seq_length - 1),
        and the dimensionality of the feature vectors dim.

        Args:
            sample: Tuple with feature dictionary and particle type. Possible features

                - "abs_pos" (N, K+1, dim), absolute positions
                - "vel_hist" (N, K*dim), historical velocity sequence
                - "vel_mag" (N,), velocity magnitudes
                - "bound" (N, 2*dim), distance to boundaries
                - "force" (N, dim), external force field
                - "rel_disp" (E, dim), relative displacement vectors
                - "rel_dist" (E, 1), relative distances, i.e. magnitude of displacements
                - "senders" (E), sender indices
                - "receivers" (E), receiver indices
        Returns:
            Dict with model output.
            The keys must be at least one of the following:

                - "acc" (N, dim), (normalized) acceleration
                - "vel" (N, dim), (normalized) velocity
                - "pos" (N, dim), (absolute) next position
        """
        raise NotImplementedError
