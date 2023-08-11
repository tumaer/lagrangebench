"""Default lagrangebench values"""

from dataclasses import dataclass, field
from typing import List

import jax.numpy as jnp

from lagrangebench.utils import PushforwardConfig


@dataclass(frozen=True)
class defaults:
    # training
    seed: int = 0
    batch_size: int = 1
    step_max: int = 1e7
    dtype: jnp.dtype = jnp.float32
    magnitude_features: bool = False
    isotropic_norm: bool = False

    # learning rate
    lr_start: float = 1e-4
    lr_end: float = 1e-6
    lr_steps: int = 5e6
    lr_decay_rate: float = 0.1

    # training tricks
    noise_std: float = 1e-4
    pushforward: PushforwardConfig = PushforwardConfig()

    # evaluation
    input_seq_length: int = 6
    # TODO make this a dataset parameter
    n_rollout_steps: int = -1
    eval_n_trajs: int = 1
    rollout_dir: str = None
    out_type: str = None
    n_extrap_steps: int = 0
    eval_steps: int = 5000
    metrics: List = field(default_factory=lambda: ["mse"])

    # logging
    log_steps: int = 1000

    # neighbor list
    neighbor_list_backend: str = "jaxmd_vmap"
    neighbor_list_multiplier: float = 1.25
