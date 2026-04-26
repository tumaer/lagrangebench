"""General utils and config structures."""

import enum
import json
import os
import random
from typing import Callable, Tuple

import cloudpickle
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import torch


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
    res = jnp.logical_or(res, particle_type == NodeType.PAD_VALUE)
    return res


def broadcast_to_batch(sample, batch_size: int):
    """Broadcast a pytree to a batched one with first dimension ``batch_size``."""
    assert batch_size > 0
    return jax.tree.map(
        lambda x: jnp.repeat(x[None, ...], batch_size, axis=0), sample
    )


def broadcast_from_batch(batch, index: int):
    """Select ``index``th sample from a batched pytree."""
    assert index >= 0
    return jax.tree.map(lambda x: x[index], batch)


def save_model(ckp_dir: str, model: eqx.Module, opt_state, metadata_ckp) -> None:
    """Save model, optimizer state and metadata to ckp_dir.

    Additionally it tracks and saves the best model to ckp_dir/best.

    Serializes the model's leaf arrays with :func:`equinox.tree_serialise_leaves`.
    The *structure* of the model is not stored; loading requires an instance of
    the same ``eqx.Module`` class with identical hyperparameters.
    """
    os.makedirs(ckp_dir, exist_ok=True)
    eqx.tree_serialise_leaves(os.path.join(ckp_dir, "model.eqx"), model)
    with open(os.path.join(ckp_dir, "opt_state.pkl"), "wb") as f:
        cloudpickle.dump(opt_state, f)
    with open(os.path.join(ckp_dir, "metadata_ckp.json"), "w") as f:
        json.dump(metadata_ckp, f)

    if "best" not in ckp_dir:
        ckp_dir_best = os.path.join(ckp_dir, "best")
        metadata_best_path = os.path.join(ckp_dir, "best", "metadata_ckp.json")
        tag = ""
        if os.path.exists(metadata_best_path):
            with open(metadata_best_path, "r") as fp:
                metadata_ckp_best = json.loads(fp.read())
            if metadata_ckp["loss"] is not None and (
                metadata_ckp_best["loss"] is None
                or metadata_ckp["loss"] < metadata_ckp_best["loss"]
            ):
                save_model(ckp_dir_best, model, opt_state, metadata_ckp)
                tag = " (best so far)"
        else:
            save_model(ckp_dir_best, model, opt_state, metadata_ckp)

        print(
            f"saved model to {ckp_dir} at step {metadata_ckp['step']}"
            f" with loss {metadata_ckp['loss']}{tag}"
        )


def load_model(model_dir: str, template: eqx.Module):
    """Load model, optimizer state and last training step from model_dir.

    The ``template`` argument must be an ``eqx.Module`` instance with the same
    architecture as the saved one; its arrays will be replaced with the
    deserialized leaves.

    Returns ``(model, opt_state, step)``.
    """
    model = eqx.tree_deserialise_leaves(os.path.join(model_dir, "model.eqx"), template)
    opt_state_path = os.path.join(model_dir, "opt_state.pkl")
    if os.path.exists(opt_state_path):
        with open(opt_state_path, "rb") as f:
            opt_state = cloudpickle.load(f)
    else:
        opt_state = None
    with open(os.path.join(model_dir, "metadata_ckp.json"), "r") as fp:
        metadata_ckp = json.loads(fp.read())
    print(f"Loaded model from {model_dir} at step {metadata_ckp['step']}")
    return model, opt_state, metadata_ckp["step"]


def get_num_params(model_or_tree):
    """Get the number of parameters in an Equinox model (or any pytree)."""
    return sum(
        np.prod(p.shape)
        for p in jax.tree.leaves(eqx.filter(model_or_tree, eqx.is_inexact_array))
    )


def set_seed(seed: int) -> Tuple[jax.Array, Callable, torch.Generator]:
    """Set seeds for jax, random and torch."""
    key = jax.random.PRNGKey(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    def seed_worker(_):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    generator = torch.Generator()
    generator.manual_seed(seed)
    return key, seed_worker, generator
