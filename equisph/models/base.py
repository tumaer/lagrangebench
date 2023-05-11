from abc import ABC, abstractmethod
from argparse import Namespace
from typing import Dict, Tuple, Type

import haiku as hk
import jax.numpy as jnp


class BaseModel(hk.Module, ABC):
    @abstractmethod
    def __call__(
        self, sample: Tuple[Dict[str, jnp.ndarray], jnp.ndarray]
    ) -> jnp.ndarray:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def setup_model(cls, args: Namespace) -> Tuple["BaseModel", Type]:
        """Setup model based on args."""
        raise NotImplementedError
