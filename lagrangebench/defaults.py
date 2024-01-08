"""Default lagrangebench values."""

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class defaults:
    """
    Default lagrangebench values.

    Attributes:
        seed: random seed. Default 0.
        batch_size: batch size. Default 1.
        step_max: max number of training steps. Default ``1e7``.
        dtype: data type. Default ``jnp.float32``.
        magnitude_features: whether to include velocity magnitudes. Default False.
        isotropic_norm: whether to normalize dimensions equally. Default False.
        lr_start: initial learning rate. Default 1e-4.
        lr_final: final learning rate (after exponential decay). Default 1e-6.
        lr_decay_steps: number of steps to decay learning rate
        lr_decay_rate: learning rate decay rate. Default 0.1.
        noise_std: standard deviation of the GNS-style noise. Default 1e-4.
        input_seq_length: number of input steps. Default 6.
        n_rollout_steps: number of eval rollout steps. -1 is full rollout. Default -1.
        eval_n_trajs: number of trajectories to evaluate. Default 1 trajectory.
        rollout_dir: directory to save rollouts. Default None.
        out_type: type of output. None means no rollout is stored. Default None.
        n_extrap_steps: number of extrapolation steps. Default 0.
        log_steps: number of steps between logs. Default 1000.
        eval_steps: number of steps between evaluations and checkpoints. Default 5000.
        neighbor_list_backend: neighbor list routine. Default "jaxmd_vmap".
        neighbor_list_multiplier: multiplier for neighbor list capacity. Default 1.25.
    """

    # training
    seed: int = 0  # random seed
    batch_size: int = 1  # batch size
    step_max: int = 5e5  # max number of training steps
    dtype: jnp.dtype = jnp.float64  # data type for preprocessing
    magnitude_features: bool = False  # whether to include velocity magnitude features
    isotropic_norm: bool = False  #  whether to normalize dimensions equally
    num_workers: int = 4  # number of workers for data loading

    # learning rate
    lr_start: float = 1e-4  # initial learning rate
    lr_final: float = 1e-6  # final learning rate (after exponential decay)
    lr_decay_steps: int = 1e5  # number of steps to decay learning rate
    lr_decay_rate: float = 0.1  # learning rate decay rate

    noise_std: float = 3e-4  # standard deviation of the GNS-style noise

    # evaluation
    input_seq_length: int = 6  # number of input steps
    n_rollout_steps: int = -1  # number of eval rollout steps. -1 is full rollout
    eval_n_trajs: int = 1  # number of trajectories to evaluate
    rollout_dir: str = None  # directory to save rollouts
    out_type: str = "none"  # type of output. None means no rollout is stored
    n_extrap_steps: int = 0  # number of extrapolation steps
    metrics_stride: int = 10  # stride for e_kin and sinkhorn
    batch_size_infer: int = 2  # batch size for validation/testing

    # logging
    log_steps: int = 1000  # number of steps between logs
    eval_steps: int = 10000  # number of steps between evaluations and checkpoints

    # neighbor list
    neighbor_list_backend: str = "jaxmd_vmap"  # backend for neighbor list computation
    neighbor_list_multiplier: float = 1.25  # multiplier for neighbor list capacity
