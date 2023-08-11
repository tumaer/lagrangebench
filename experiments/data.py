import json
import os
import os.path as osp
from argparse import Namespace
from typing import Callable, Dict, Tuple

import jax.numpy as jnp

from lagrangebench.data import H5Dataset


def setup_data(args: Namespace) -> Tuple[Dict, H5Dataset, H5Dataset, Callable]:
    if not osp.isabs(args.config.data_dir):
        args.config.data_dir = osp.join(os.getcwd(), args.config.data_dir)

    args.info.dataset_name = osp.basename(args.config.data_dir.split("/")[-1])
    if args.config.ckp_dir is not None:
        os.makedirs(args.config.ckp_dir, exist_ok=True)
    if args.config.rollout_dir is not None:
        os.makedirs(args.config.rollout_dir, exist_ok=True)
    with open(osp.join(args.config.data_dir, "metadata.json"), "r") as f:
        metadata = json.loads(f.read())

    # dataloader
    train_seq_l = args.config.input_seq_length + args.config.pushforward["unrolls"][-1]
    data_train = H5Dataset(
        args.config.data_dir,
        "train",
        metadata=metadata,
        input_seq_length=train_seq_l,
        is_rollout=False,
        nl_backend=args.config.neighbor_list_backend,
        name=args.info.dataset_name,
    )
    data_eval = H5Dataset(
        dataset_path=args.config.data_dir,
        split="test" if args.config.test else "valid",
        metadata=metadata,
        input_seq_length=args.config.input_seq_length,
        split_valid_traj_into_n=args.config.split_valid_traj_into_n,
        is_rollout=True,
        nl_backend=args.config.neighbor_list_backend,
        name=args.info.dataset_name,
    )

    if args.config.n_rollout_steps == -1:
        args.config.n_rollout_steps = (
            data_eval.subsequence_length - args.config.input_seq_length
        )

    # TODO move external force to dataset
    if "RPF" in args.info.dataset_name.upper():
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

    else:
        args.info.has_external_force = False
        external_force_fn = None

    return metadata, data_train, data_eval, external_force_fn
