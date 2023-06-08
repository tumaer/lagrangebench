import json
import os
import pickle
import random
from dataclasses import dataclass
from typing import Callable, Tuple

import cloudpickle
import jax
import jax.numpy as jnp
import numpy as np
import pyvista
import torch


def broadcast_to_batch(sample, batch_size: int):
    """Broadcast a pytree to a batched one with first dimension batch_size"""
    assert batch_size > 0
    return jax.tree_map(lambda x: jnp.repeat(x[None, ...], batch_size, axis=0), sample)


def broadcast_from_batch(batch, index: int):
    """Broadcast a batched pytree to the sample `index` out of the batch"""
    assert index >= 0
    return jax.tree_map(lambda x: x[index], batch)


def save_pytree(ckp_dir: str, pytree_obj, name) -> None:
    with open(os.path.join(ckp_dir, f"{name}_array.npy"), "wb") as f:
        for x in jax.tree_leaves(pytree_obj):
            np.save(f, x, allow_pickle=False)

    tree_struct = jax.tree_map(lambda t: 0, pytree_obj)
    with open(os.path.join(ckp_dir, f"{name}_tree.pkl"), "wb") as f:
        pickle.dump(tree_struct, f)


def save_haiku(ckp_dir: str, params, state, opt_state, metadata_ckp) -> None:
    """https://github.com/deepmind/dm-haiku/issues/18"""

    save_pytree(ckp_dir, params, "params")
    save_pytree(ckp_dir, state, "state")

    with open(os.path.join(ckp_dir, "opt_state.pkl"), "wb") as f:
        cloudpickle.dump(opt_state, f)
    with open(os.path.join(ckp_dir, "metadata_ckp.json"), "w") as f:
        json.dump(metadata_ckp, f)

    if "best" not in ckp_dir:
        ckp_dir_best = os.path.join(ckp_dir, "best")
        metadata_best_path = os.path.join(ckp_dir, "best", "metadata_ckp.json")

        if os.path.exists(metadata_best_path):  # all except first step
            with open(metadata_best_path, "r") as fp:
                metadata_ckp_best = json.loads(fp.read())

            # if loss is better than best previous loss, save to best model directory
            if metadata_ckp["loss"] < metadata_ckp_best["loss"]:
                print(
                    f"Saving model to {ckp_dir} at step {metadata_ckp['step']}"
                    f" with loss {metadata_ckp['loss']} (best so far)"
                )

                save_haiku(ckp_dir_best, params, state, opt_state, metadata_ckp)
        else:  # first step
            save_haiku(ckp_dir_best, params, state, opt_state, metadata_ckp)


def load_pytree(model_dir: str, name):
    """load a pytree from a directory"""
    with open(os.path.join(model_dir, f"{name}_tree.pkl"), "rb") as f:
        tree_struct = pickle.load(f)

    leaves, treedef = jax.tree_flatten(tree_struct)

    with open(os.path.join(model_dir, f"{name}_array.npy"), "rb") as f:
        flat_state = [np.load(f) for _ in leaves]

    return jax.tree_unflatten(treedef, flat_state)


def load_haiku(model_dir: str):
    """https://github.com/deepmind/dm-haiku/issues/18"""

    print("Loading model from", model_dir)

    params = load_pytree(model_dir, "params")
    state = load_pytree(model_dir, "state")

    with open(os.path.join(model_dir, "opt_state.pkl"), "rb") as f:
        opt_state = cloudpickle.load(f)

    with open(os.path.join(model_dir, "metadata_ckp.json"), "r") as fp:
        metadata_ckp = json.loads(fp.read())

    return params, state, opt_state, metadata_ckp["step"]


def get_num_params(params):
    """Get the number of parameters in a Haiku model"""
    return sum(np.prod(p.shape) for p in jax.tree_leaves(params))


def print_params_shapes(params, prefix=""):
    if not isinstance(params, dict):
        print(f"{prefix: <40}, shape = {params.shape}")
    else:
        for k, v in params.items():
            print_params_shapes(v, prefix=prefix + k)


def write_vtk(data_dict, path):
    """Store a .vtk file for ParaView"""
    r = np.asarray(data_dict["r"])
    N, dim = r.shape

    # PyVista treats the position information differently than the rest
    if dim == 2:
        r = np.hstack([r, np.zeros((N, 1))])
    data_pv = pyvista.PolyData(r)

    # copy all the other information also to pyvista, using plain numpy arrays
    for k, v in data_dict.items():
        # skip r because we already considered it above
        if k == "r":
            continue

        # working in 3D or scalar features do not require special care
        if dim == 2 and v.ndim == 2:
            v = np.hstack([v, np.zeros((N, 1))])

        data_pv[k] = np.asarray(v)

    data_pv.save(path)


def set_seed(seed: int) -> Tuple[jax.random.KeyArray, Callable, torch.Generator]:
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
class LossWeights:
    """Weights for the different targets in the loss function"""

    pos: float = 0.0
    vel: float = 0.0
    acc: float = 0.0

    def __getitem__(self, item):
        return getattr(self, item)

    @property
    def nonzero(self):
        return [field for field in self.__annotations__ if self[field] != 0]
