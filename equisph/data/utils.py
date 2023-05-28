import json
import os
import warnings
from argparse import Namespace
from typing import Callable, Dict, List, Tuple

import jax.numpy as jnp
import numpy as np
from torch.utils.data import DataLoader

from .data import H5Dataset


def get_dataset_stats(
    metadata: Dict[str, List[float]],
    is_isotropic_norm: bool,
    noise_std: float,
) -> Dict[str, Dict[str, jnp.ndarray]]:
    acc_mean = jnp.array(metadata["acc_mean"])
    acc_std = jnp.array(metadata["acc_std"])
    vel_mean = jnp.array(metadata["vel_mean"])
    vel_std = jnp.array(metadata["vel_std"])

    if is_isotropic_norm:
        warnings.warn(
            "The isotropic normalization is only a simplification of the general case."
            "It is only valid if the means of the velocity and acceleration are"
            "isotropic -> we use $max(abs(mean)) < 1% min(std)$ as a heuristic."
        )

        acc_mean = jnp.mean(acc_mean) * jnp.ones_like(acc_mean)
        acc_std = jnp.sqrt(jnp.mean(acc_std**2)) * jnp.ones_like(acc_std)
        vel_mean = jnp.mean(vel_mean) * jnp.ones_like(vel_mean)
        vel_std = jnp.sqrt(jnp.mean(vel_std**2)) * jnp.ones_like(vel_std)

    return {
        "acceleration": {
            "mean": acc_mean,
            "std": jnp.sqrt(acc_std**2 + noise_std**2),
        },
        "velocity": {
            "mean": vel_mean,
            "std": jnp.sqrt(vel_std**2 + noise_std**2),
        },
    }


def numpy_collate(batch) -> np.ndarray:
    # NOTE: to numpy to avoid copying twice (dataloader timeout).
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    if isinstance(batch[0], (tuple, list)):
        return type(batch[0])(numpy_collate(samples) for samples in zip(*batch))
    else:
        return np.asarray(batch)


def setup_data(
    args: Namespace, seed_worker, generator
) -> Tuple[Dict, DataLoader, DataLoader, Callable]:
    if not os.path.isabs(args.config.data_dir):
        args.config.data_dir = os.path.join(os.getcwd(), args.config.data_dir)

    args.info.dataset_name = os.path.basename(args.config.data_dir.split("/")[-1])
    if args.config.ckp_dir is not None:
        os.makedirs(args.config.ckp_dir, exist_ok=True)
    if args.config.rollout_dir is not None:
        os.makedirs(args.config.rollout_dir, exist_ok=True)
    with open(os.path.join(args.config.data_dir, "metadata.json"), "r") as f:
        metadata = json.loads(f.read())

    # dataloader
    train_seq_l = args.config.input_seq_length + args.config.pushforward["unrolls"][-1]
    data_train = H5Dataset(
        args.config.data_dir,
        "train",
        args.config.perc_train,
        train_seq_l,
        is_rollout=False,
    )
    loader_train = DataLoader(
        dataset=data_train,
        batch_size=args.config.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=numpy_collate,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    data_eval = H5Dataset(
        dataset_path=args.config.data_dir,
        split="test" if args.config.test else "valid",
        input_seq_length=args.config.input_seq_length,
        split_valid_traj_into_n=args.config.split_valid_traj_into_n,
        is_rollout=True,
    )
    loader_eval = DataLoader(
        dataset=data_eval,
        batch_size=1,
        collate_fn=numpy_collate,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    assert (
        args.config.n_rollout_steps
        <= data_eval.subsequence_length - args.config.input_seq_length
    ), (
        "If you want to evaluate the loss on more than the ground truth traj length, "
        "then use the --eval_n_more_steps argument."
    )
    assert args.config.eval_n_trajs <= len(
        loader_eval
    ), "eval_n_trajs must be <= len(loader_valid)"

    if args.config.n_rollout_steps == -1:
        args.config.n_rollout_steps = (
            data_eval.subsequence_length - args.config.input_seq_length
        )

    if "TGV" in args.info.dataset_name.upper():
        args.info.has_external_force = False
        external_force_fn = None
    elif "RPF" in args.info.dataset_name:
        args.info.has_external_force = True
        if metadata["dim"] == 2:

            def external_force_fn(position):
                return jnp.where(
                    position[1] > 1.0,
                    jnp.array([-1.0, 0.0]),
                    jnp.array([1.0, 0.0]),
                )

        elif metadata["dim"] == 3:

            def external_force_fn(position):
                return jnp.where(
                    position[1] > 1.0,
                    jnp.array([-1.0, 0.0, 0.0]),
                    jnp.array([1.0, 0.0, 0.0]),
                )

    elif "Hook" in args.info.dataset_name:
        args.info.has_external_force = False
        external_force_fn = None

    return metadata, loader_train, loader_eval, external_force_fn
