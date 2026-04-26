"""Base class for LagrangeBench models (Equinox)."""

from abc import abstractmethod
from typing import Dict, Tuple

import equinox as eqx
import jax.numpy as jnp


class BaseModel(eqx.Module):
    """Abstract base class for particle simulation models.

    Subclasses inherit from :class:`equinox.Module`. Parameters are stored as
    fields of the module; calling ``model(sample)`` performs a forward pass.

    We specify the dimensions of the inputs and outputs using the number of nodes
    ``N``, the number of edges ``E``, the number of historic velocities
    ``K = input_seq_length - 1``, and the feature dimensionality ``dim``.

    Expected ``features`` dict entries (present ones depend on the dataset):

    - ``abs_pos`` ``(N, K+1, dim)`` — absolute positions
    - ``vel_hist`` ``(N, K*dim)`` — historical velocity sequence
    - ``vel_mag`` ``(N,)`` — velocity magnitudes
    - ``bound`` ``(N, 2*dim)`` — distance to boundaries
    - ``force`` ``(N, dim)`` — external force field
    - ``rel_disp`` ``(E, dim)`` — relative displacement vectors
    - ``rel_dist`` ``(E, 1)`` — relative distances
    - ``senders``, ``receivers`` ``(E,)`` — edge indices

    Return dict with at least one of:

    - ``acc`` ``(N, dim)`` — (normalized) acceleration
    - ``vel`` ``(N, dim)`` — (normalized) velocity
    - ``pos`` ``(N, dim)`` — absolute next position
    """

    @abstractmethod
    def __call__(
        self, sample: Tuple[Dict[str, jnp.ndarray], jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        raise NotImplementedError
