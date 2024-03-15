"""General utils and config structures."""

import enum
import json
import math
import os
import pickle
import random
from dataclasses import dataclass, field
from typing import Callable, List, Tuple

import cloudpickle
import jax
import jax.numpy as jnp
import numpy as np
import torch


# TODO look for a better place to put this and get_kinematic_mask
class NodeType(enum.IntEnum):
    """Particle types."""

    PAD_VALUE = -1
    FLUID = 0
    SOLID_WALL = 1
    MOVING_WALL = 2
    RIGID_BODY = 3
    SIZE = 9


def get_kinematic_mask(particle_type):
    """Return a boolean mask, set to true for all kinematic (obstacle) particles."""
    res = jnp.logical_or(
        particle_type == NodeType.SOLID_WALL, particle_type == NodeType.MOVING_WALL
    )
    # In datasets with variable number of particles we treat padding as kinematic nodes
    res = jnp.logical_or(res, particle_type == NodeType.PAD_VALUE)
    return res


def broadcast_to_batch(sample, batch_size: int):
    """Broadcast a pytree to a batched one with first dimension batch_size."""
    assert batch_size > 0
    return jax.tree_map(lambda x: jnp.repeat(x[None, ...], batch_size, axis=0), sample)


def broadcast_from_batch(batch, index: int):
    """Broadcast a batched pytree to the sample `index` out of the batch."""
    assert index >= 0
    return jax.tree_map(lambda x: x[index], batch)


def save_pytree(ckp_dir: str, pytree_obj, name) -> None:
    """Save a pytree to a directory."""
    with open(os.path.join(ckp_dir, f"{name}_array.npy"), "wb") as f:
        for x in jax.tree_leaves(pytree_obj):
            np.save(f, x, allow_pickle=False)

    tree_struct = jax.tree_map(lambda t: 0, pytree_obj)
    with open(os.path.join(ckp_dir, f"{name}_tree.pkl"), "wb") as f:
        pickle.dump(tree_struct, f)


def save_haiku(ckp_dir: str, params, state, opt_state, metadata_ckp) -> None:
    """Save params, state and optimizer state to ckp_dir.

    Additionally it tracks and saves the best model to ckp_dir/best.

    See: https://github.com/deepmind/dm-haiku/issues/18
    """
    save_pytree(ckp_dir, params, "params")
    save_pytree(ckp_dir, state, "state")

    with open(os.path.join(ckp_dir, "opt_state.pkl"), "wb") as f:
        cloudpickle.dump(opt_state, f)
    with open(os.path.join(ckp_dir, "metadata_ckp.json"), "w") as f:
        json.dump(metadata_ckp, f)

    # only run for the main checkpoint directory (not best)
    if "best" not in ckp_dir:
        ckp_dir_best = os.path.join(ckp_dir, "best")
        metadata_best_path = os.path.join(ckp_dir, "best", "metadata_ckp.json")
        tag = ""

        if os.path.exists(metadata_best_path):  # all except first step
            with open(metadata_best_path, "r") as fp:
                metadata_ckp_best = json.loads(fp.read())

            # if loss is better than best previous loss, save to best model directory
            if metadata_ckp["loss"] < metadata_ckp_best["loss"]:
                save_haiku(ckp_dir_best, params, state, opt_state, metadata_ckp)
                tag = " (best so far)"
        else:  # first step
            save_haiku(ckp_dir_best, params, state, opt_state, metadata_ckp)

        print(
            f"saved model to {ckp_dir} at step {metadata_ckp['step']}"
            f" with loss {metadata_ckp['loss']}{tag}"
        )


def load_pytree(model_dir: str, name):
    """Load a pytree from a directory."""
    with open(os.path.join(model_dir, f"{name}_tree.pkl"), "rb") as f:
        tree_struct = pickle.load(f)

    leaves, treedef = jax.tree_flatten(tree_struct)

    with open(os.path.join(model_dir, f"{name}_array.npy"), "rb") as f:
        flat_state = [np.load(f) for _ in leaves]

    return jax.tree_unflatten(treedef, flat_state)


def load_haiku(model_dir: str):
    """Load params, state, optimizer state and last training step from model_dir.

    See: https://github.com/deepmind/dm-haiku/issues/18
    """
    params = load_pytree(model_dir, "params")
    state = load_pytree(model_dir, "state")

    with open(os.path.join(model_dir, "opt_state.pkl"), "rb") as f:
        opt_state = cloudpickle.load(f)

    with open(os.path.join(model_dir, "metadata_ckp.json"), "r") as fp:
        metadata_ckp = json.loads(fp.read())

    print(f"Loaded model from {model_dir} at step {metadata_ckp['step']}")

    return params, state, opt_state, metadata_ckp["step"]


def get_num_params(params):
    """Get the number of parameters in a Haiku model."""
    return sum(np.prod(p.shape) for p in jax.tree_leaves(params))


def print_params_shapes(params, prefix=""):
    if not isinstance(params, dict):
        print(f"{prefix: <40}, shape = {params.shape}")
    else:
        for k, v in params.items():
            print_params_shapes(v, prefix=prefix + k)


def set_seed(seed: int) -> Tuple[jax.Array, Callable, torch.Generator]:
    """Set seeds for jax, random and torch."""
    # first PRNG key
    key = jax.random.PRNGKey(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # dataloader-related seeds
    def seed_worker(_):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return key, seed_worker, generator


@dataclass(frozen=True)
class LossConfig:
    """Weights for the different targets in the loss function."""

    pos: float = 0.0
    vel: float = 0.0
    acc: float = 1.0
    noise: float = 0.0

    def __getitem__(self, item):
        return getattr(self, item)

    @property
    def nonzero(self):
        return [field for field in self.__annotations__ if self[field] != 0]


@dataclass(frozen=False)
class PushforwardConfig:
    """Pushforward trick configuration.

    Attributes:
        steps: When to introduce each unroll stage, e.g. [-1, 20000, 50000]
        unrolls: For how many timesteps to unroll, e.g. [0, 1, 20]
        probs: Probability (ratio) between the relative unrolls, e.g. [5, 4, 1]
    """

    steps: List[int] = field(default_factory=lambda: [-1])
    unrolls: List[int] = field(default_factory=lambda: [0])
    probs: List[float] = field(default_factory=lambda: [1.0])

    def __getitem__(self, item):
        return getattr(self, item)


# For PDE Refiner
def fourier_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings.

    Args:
        timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
        dim (int): the dimension of the output.
        max_period (int): controls the minimum frequency of the embeddings.

    Returns:
        embedding (jax.numpy.DeviceArray): [N x dim] array of positional embeddings.
    """
    half = dim // 2
    freqs = jnp.exp(
        -math.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half
    )
    args = timesteps[:, None].astype(jnp.float32) * freqs[None, :]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2:  # enters the 'if-statement' if the dim is odd
        embedding = jnp.concatenate(
            [embedding, jnp.zeros((embedding.shape[0], 1), dtype=embedding.dtype)],
            axis=-1,
        )
    return embedding


# For ACDM
def linear_beta_schedule(timesteps):
    # if timesteps < 10:
    #     raise ValueError(
    #         "Warning: Less than 10 timesteps require adjustments \
    #         to this schedule!"
    #     )

    beta_start = 0.0001 * (
        500 / timesteps
    )  # adjust reference values determined for 500 steps
    beta_end = 0.02 * (500 / timesteps)
    betas = jnp.linspace(beta_start, beta_end, timesteps)
    return jnp.clip(betas, 0.0001, 0.9999)


class ACDMConfig:
    def __init__(self, diffusionSteps, num_conditioning_steps, conditioning_parameter):
        # For Diffusion
        self.diffusionSteps = diffusionSteps
        self.num_conditioning_steps = num_conditioning_steps
        self.conditioning_parameter = conditioning_parameter

        self.betas = linear_beta_schedule(timesteps=self.diffusionSteps)
        self.alphas = 1.0 - self.betas
        self.alphasCumprod = jnp.cumprod(self.alphas, axis=0)

        # Here we create a padding array of ones and concatenate it with alphasCumprod
        pad = jnp.ones((1,) + self.alphasCumprod.shape[1:])
        self.alphasCumprodPrev = jnp.concatenate([pad, self.alphasCumprod[:-1]], axis=0)

        self.sqrtRecipAlphas = jnp.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrtAlphasCumprod = jnp.sqrt(self.alphasCumprod)
        self.sqrtOneMinusAlphasCumprod = jnp.sqrt(1.0 - self.alphasCumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0) (ONLY REQ FOR INFERENCE)
        self.posteriorVariance = (
            self.betas * (1.0 - self.alphasCumprodPrev) / (1.0 - self.alphasCumprod)
        )
        self.sqrtPosteriorVariance = jnp.sqrt(self.posteriorVariance)

    def __getitem__(self, item):
        return getattr(self, item)
