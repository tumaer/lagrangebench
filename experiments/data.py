import os
import os.path as osp
from argparse import Namespace
from typing import Callable, Tuple

import jax.numpy as jnp

from lagrangebench.data import H5Dataset


def setup_data(args: Namespace) -> Tuple[H5Dataset, H5Dataset, Callable]:
    if not osp.isabs(args.config.data_dir):
        args.config.data_dir = osp.join(os.getcwd(), args.config.data_dir)

    args.info.dataset_name = osp.basename(args.config.data_dir.split("/")[-1])
    if args.config.ckp_dir is not None:
        os.makedirs(args.config.ckp_dir, exist_ok=True)
    if args.config.rollout_dir is not None:
        os.makedirs(args.config.rollout_dir, exist_ok=True)

    # dataloader
    train_seq_l = args.config.input_seq_length + args.config.pushforward["unrolls"][-1]
    data_train = H5Dataset(
        "train",
        dataset_path=args.config.data_dir,
        input_seq_length=train_seq_l,
        is_rollout=False,
        nl_backend=args.config.neighbor_list_backend,
    )
    data_eval = H5Dataset(
        "test" if args.config.test else "valid",
        dataset_path=args.config.data_dir,
        input_seq_length=args.config.input_seq_length,
        split_valid_traj_into_n=args.config.split_valid_traj_into_n,
        is_rollout=True,
        nl_backend=args.config.neighbor_list_backend,
    )

    if args.config.n_rollout_steps == -1:
        args.config.n_rollout_steps = (
            data_eval.subsequence_length - args.config.input_seq_length
        )

    if "RPF" in args.info.dataset_name.upper():
        args.info.has_external_force = True
        if data_train.metadata["dim"] == 2:

            def external_force_fn(position):
                return jnp.where(
                    position[1] > 1.0,
                    jnp.array([-1.0, 0.0]),
                    jnp.array([1.0, 0.0]),
                )

        elif data_train.metadata["dim"] == 3:

            def external_force_fn(position):
                return jnp.where(
                    position[1] > 1.0,
                    jnp.array([-1.0, 0.0, 0.0]),
                    jnp.array([1.0, 0.0, 0.0]),
                )

    else:
        args.info.has_external_force = False
        external_force_fn = None

    data_train.external_force_fn = external_force_fn
    data_eval.external_force_fn = external_force_fn

    return data_train, data_eval
