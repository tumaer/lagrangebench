"""GNS in JAX with Haiku, Jraph, and JAX-MD"""

import argparse
import json
import os
from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax
from jax import vmap
from torch.utils.data import DataLoader

import wandb
from gns_jax.data import H5Dataset, numpy_collate
from gns_jax.gns import GNS
from gns_jax.utils import (
    NodeType,
    broadcast_from_batch,
    broadcast_to_batch,
    eval_rollout,
    get_kinematic_mask,
    get_num_params,
    load_haiku,
    save_haiku,
    setup_builder,
)


def train(model, params, state, neighbors, loader_train, loader_valid, setup, args):

    # checkpointing and logging
    run_prefix = "_".join(["gns", args.dataset_name, ""])
    i = 0
    while os.path.isdir(os.path.join(args.ckp_dir, run_prefix + str(i))):
        i += 1
    run_name = run_prefix + str(i)
    ckp_dir = os.path.join(args.ckp_dir, run_name)
    os.makedirs(ckp_dir, exist_ok=True)

    if args.wandb:
        wandb.init(
            project="segnn",
            entity="segnn-sph",
            name=run_name,
            config=args,
            save_code=True,
        )

    # Set learning rate to decay from 1e-4 to 1e-6 over 10M steps exponentially
    # and then keep it at 1e-6
    lr_scheduler = optax.exponential_decay(
        init_value=args.lr_start,
        transition_steps=int(5e6),
        decay_rate=args.lr_decay_rate,
        end_value=args.lr_final,
    )
    opt_init, opt_update = optax.adam(learning_rate=lr_scheduler)
    opt_state = opt_init(params)

    # Precompile model for evaluation
    model_apply = jax.jit(model.apply)

    def loss_fn(
        params: hk.Params,
        state: hk.State,
        graph: jraph.GraphsTuple,
        target: jnp.ndarray,
        particle_type: jnp.ndarray,
    ):
        pred, state = model.apply(params, state, (graph, particle_type))
        assert target.shape == pred.shape

        non_kinematic_mask = jnp.logical_not(get_kinematic_mask(particle_type))
        num_non_kinematic = non_kinematic_mask.sum()
        # MSE loss
        loss = ((pred - target) ** 2).sum(axis=-1)
        loss = jnp.where(non_kinematic_mask, loss, 0)
        loss = loss.sum() / num_non_kinematic

        return loss, state

    @jax.jit
    def train_step(
        params: hk.Module,
        state: hk.State,
        graph_batch: Tuple[jraph.GraphsTuple, ...],
        target_batch: Tuple[jnp.ndarray, ...],
        particle_type_batch: Tuple[jnp.ndarray, ...],
        opt_state: optax.OptState,
    ) -> Tuple[float, hk.Params, hk.State, optax.OptState]:

        value_and_grad_vmap = vmap(
            jax.value_and_grad(loss_fn, has_aux=True), in_axes=(None, None, 0, 0, 0)
        )
        (loss, state), grads = value_and_grad_vmap(
            params, state, graph_batch, target_batch, particle_type_batch
        )

        # aggregate over the first (batch) dimension of each leave element
        grads = jax.tree_map(lambda x: x.sum(axis=0), grads)
        state = jax.tree_map(lambda x: x.sum(axis=0), state)
        loss = jax.tree_map(lambda x: x.mean(axis=0), loss)

        updates, opt_state = opt_update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return loss, new_params, state, opt_state

    preprocess_vmap = jax.vmap(setup.preprocess, in_axes=(0, 0, None, 0))

    # prepare for batch training.
    key = jax.random.PRNGKey(args.seed)
    keys = jax.random.split(key, args.batch_size)
    neighbors_batch = broadcast_to_batch(neighbors, args.batch_size)

    step_digits = len(str(int(args.step_max)))
    step = 0
    while step < args.step_max:

        for raw_batch in loader_train:
            # pos_input, particle_type, pos_target = raw_batch

            keys, graph_batch, target_batch, neighbors_batch = preprocess_vmap(
                keys, raw_batch, args.noise_std, neighbors_batch
            )

            if neighbors_batch.did_buffer_overflow.sum() > 0:
                # check if the neighbor list is too small for any of the
                # samples in the batch. If so, reallocate the neighbor list
                ind = np.argmax(neighbors_batch.did_buffer_overflow)
                edges_ = neighbors_batch.idx[ind].shape
                print(f"Reallocate neighbors list {edges_} at step {step}")
                sample = broadcast_from_batch(raw_batch, index=ind)
                _, _, _, nbrs = setup.allocate(keys[0], sample)
                print(f"To list {nbrs.idx.shape}")

                neighbors_batch = broadcast_to_batch(nbrs, args.batch_size)

                # To run the loop N times even if sometimes
                # did_buffer_overflow > 0 we directly return to the beginning
                continue

            loss, params, state, opt_state = train_step(
                params=params,
                state=state,
                graph_batch=graph_batch,
                target_batch=target_batch,
                particle_type_batch=raw_batch[1],
                opt_state=opt_state,
            )

            if step % args.log_steps == 0:
                if args.wandb:
                    wandb.log({"train/loss": loss.item()}, step)
                else:
                    step_str = str(step).zfill(step_digits)
                    print(f"{step_str}, train/loss: {loss.item():.5f}.")

            if step % args.save_steps == 0:
                save_haiku(os.path.join(ckp_dir), params)

            if step % args.eval_steps == 0 and step > 0:
                nbrs = broadcast_from_batch(neighbors_batch, index=0)
                eval_loss, nbrs = eval_rollout(
                    setup=setup,
                    model_apply=model_apply,
                    params=params,
                    state=state,
                    neighbors=nbrs,
                    loader_valid=loader_valid,
                    num_rollout_steps=args.num_rollout_steps + 1,  # +1 <- not training
                    num_trajs=5,
                    save_results=False,
                    rollout_dir=args.rollout_dir,
                    is_write_vtk=args.write_vtk,
                )
                # In the beginning of training, the dynamics ae very random and
                # we don't want to influence the training neighbors list by
                # this extreme case of the dynamics. Therefore, we only update
                # the neighbors list inside of the eval_rollout function.
                # neighbors_batch = broadcast_to_batch(nbrs, args.batch_size)

                if args.wandb:
                    wandb.log({"val/loss": eval_loss}, step)
                else:
                    print(f"val/loss: {eval_loss}")

            step += 1
            if step == args.step_max:
                break


def infer(model, params, state, neighbors, loader_valid, setup, args):
    model_apply = jax.jit(model.apply)
    eval_loss, _ = eval_rollout(
        setup=setup,
        model_apply=model_apply,
        params=params,
        state=state,
        neighbors=neighbors,
        loader_valid=loader_valid,
        num_rollout_steps=args.num_rollout_steps + 1,  # +1 because we are not training
        num_trajs=5,
        save_results=True,
        rollout_dir=args.rollout_dir,
        is_write_vtk=args.write_vtk,
    )
    print(f"Rollout loss = {eval_loss}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "infer"])
    parser.add_argument("--step_max", type=int, default=2e7)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr_start", type=float, default=1e-4)
    parser.add_argument("--lr_final", type=float, default=1e-6)
    parser.add_argument("--lr_decay_rate", type=float, default=0.1)
    parser.add_argument("--lr_decay_steps", type=int, default=5e6)
    parser.add_argument("--noise_std", type=float, default=6.7e-4)
    parser.add_argument("--input_seq_length", type=int, default=6)
    parser.add_argument("--num_mp_steps", type=int, default=10)
    parser.add_argument("--latent_dim", type=int, default=128)

    parser.add_argument("--wandb", action="store_true", help="Log to wandb")
    parser.add_argument("--log_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=5000)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--model_dir", type=str, help="To load a checkpoint")
    parser.add_argument("--data_dir", type=str, help="Path to the dataset")
    parser.add_argument("--ckp_dir", type=str, default="ckp")
    parser.add_argument("--rollout_dir", type=str)
    parser.add_argument("--write_vtk", action="store_true", help="vtk rollout")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.dataset_name = os.path.basename(args.data_dir.split("/")[-1])

    if args.ckp_dir is not None:
        os.makedirs(args.ckp_dir, exist_ok=True)

    if args.rollout_dir is not None:
        os.makedirs(args.rollout_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, "metadata.json"), "r") as f:
        args.metadata = json.loads(f.read())

    args.normalization_stats = {
        "acceleration": {
            "mean": jnp.array(args.metadata["acc_mean"]),
            "std": jnp.sqrt(
                jnp.array(args.metadata["acc_std"]) ** 2 + args.noise_std**2
            ),
        },
        "velocity": {
            "mean": jnp.array(args.metadata["vel_mean"]),
            "std": jnp.sqrt(
                jnp.array(args.metadata["vel_std"]) ** 2 + args.noise_std**2
            ),
        },
    }

    args.num_rollout_steps = args.metadata["sequence_length"] - args.input_seq_length

    # dataloader
    data_train = H5Dataset(
        args.data_dir, "train", args.input_seq_length, is_rollout=False
    )
    loader_train = DataLoader(
        dataset=data_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=numpy_collate,
        drop_last=True,
    )
    data_valid = H5Dataset(
        args.data_dir, "valid", args.input_seq_length, is_rollout=True
    )
    loader_valid = DataLoader(
        dataset=data_valid, batch_size=1, collate_fn=numpy_collate
    )

    # neighbors search
    bounds = np.array(args.metadata["bounds"])
    args.box = bounds[:, 1] - bounds[:, 0]

    # dataset-specific setup
    if args.dataset_name == "BoxBath":
        particle_dimension = 3
        # node_in = 37, edge_in = 4
        args.box *= 1.2
    elif args.dataset_name in ["WaterDrop", "WaterDropSample"]:
        particle_dimension = 2
        # node_in = 30, edge_in = 3

    # preprocessing allocate and update functions in the spirit of jax-md's
    # `partition.neighbor_list`. And an integration utility with PBC.
    setup = setup_builder(args)

    # transform core simulator outputting accelerations.
    model = lambda x: GNS(
        particle_dimension=particle_dimension,
        latent_size=args.latent_dim,
        num_mlp_layers=2,
        num_message_passing_steps=args.num_mp_steps,
        num_particle_types=NodeType.SIZE,
        particle_type_embedding_size=16,
    )(x)
    # `state` is currently not used, but who knows in the future
    model = hk.without_apply_rng(hk.transform_with_state(model))

    # first PRNG key
    key = jax.random.PRNGKey(args.seed)
    # get an example to initialize the setup and model
    pos_input, particle_type, pos_target = next(iter(loader_train))
    # the torch loader give a whole batch. We take only the first sample.
    sample = (pos_input[0], particle_type[0], pos_target[0])
    # initialize setup
    key, graph, _, neighbors = setup.allocate(key, sample)
    # initialize model
    key, subkey = jax.random.split(key, 2)
    params, state = model.init(subkey, (graph, particle_type[0]))

    print(f"Model with {get_num_params(params)} parameters.")

    # load model from checkpoint if provided
    if args.model_dir:
        params = load_haiku(args.model_dir)

    if args.mode == "train":
        train(model, params, state, neighbors, loader_train, loader_valid, setup, args)
    elif args.mode == "infer":
        infer(model, params, state, neighbors, loader_valid, setup, args)
