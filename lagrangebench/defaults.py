"""Default lagrangebench values."""

from dataclasses import dataclass, field
from typing import List

import jax.numpy as jnp


@dataclass(frozen=True)
class defaults:
    """Default lagrangebench values."""

    # training
    seed: int = 0  # random seed
    batch_size: int = 1  # batch size
    step_max: int = 1e7  # max number of training steps
    dtype: jnp.dtype = jnp.float32  # data type
    magnitude_features: bool = False  # whether to include velocity magnitude features
    isotropic_norm: bool = False  # whether to use isotropic normalization

    # learning rate
    lr_start: float = 1e-4  # initial learning rate
    lr_final: float = 1e-6  # final learning rate (after exponential decay)
    lr_decay_steps: int = 5e6  # number of steps to decay learning rate
    lr_decay_rate: float = 0.1  # learning rate decay rate

    noise_std: float = 1e-4  # standard deviation of the GNS-style noise

    # evaluation
    input_seq_length: int = 6  # number of input steps
    # TODO make this a dataset parameter
    n_rollout_steps: int = -1  # number of rollout steps. -1 means full rollout
    eval_n_trajs: int = 1  # number of trajectories to evaluate
    rollout_dir: str = None  # directory to save rollouts
    out_type: str = None  # type of output. None means no rollout is stored
    n_extrap_steps: int = 0  # number of extrapolation steps
    metrics: List = field(default_factory=lambda: ["mse"])  # evaluation metrics

    # logging
    log_steps: int = 1000  # number of steps between logs
    eval_steps: int = 5000  # number of steps between evaluations and checkpoints

    # neighbor list
    neighbor_list_backend: str = "jaxmd_vmap"  # backend for neighbor list computation
    neighbor_list_multiplier: float = 1.25  # multiplier for neighbor list capacity
