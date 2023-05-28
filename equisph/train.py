import copy
import os
from argparse import Namespace
from functools import partial
from typing import Callable, Dict, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jax_md.partition as partition
import jraph
import optax
import yaml
from jax import vmap
from torch.utils.data import DataLoader

import wandb
from equisph.case_setup import CaseSetupFn, get_kinematic_mask
from equisph.evaluate import MetricsComputer, averaged_metrics, eval_rollout
from equisph.utils import (
    LossWeights,
    broadcast_from_batch,
    broadcast_to_batch,
    load_haiku,
    save_haiku,
)


def push_forward_sample_steps(key, step, pushforward):
    key, key_unroll = jax.random.split(key, 2)

    # steps needs to be an ordered list
    steps = jnp.array(pushforward["steps"])
    assert all(steps[i] <= steps[i + 1] for i in range(len(steps) - 1))

    # until which index to sample from
    idx = (step > steps).sum()

    unroll_steps = jax.random.choice(
        key_unroll,
        a=jnp.array(pushforward["unrolls"][:idx]),
        p=jnp.array(pushforward["probs"][:idx]),
    )
    return key, unroll_steps


def push_forward_build(model_apply, case):
    @jax.jit
    def push_forward(features, current_pos, particle_type, neighbors, params, state):
        # no buffer overflow check here, since push forward acts on later epochs
        pred, _ = model_apply(params, state, (features, particle_type))
        next_pos = case.integrate(pred, current_pos)
        current_pos = jnp.concatenate(
            [current_pos[:, 1:], next_pos[:, None, :]], axis=1
        )

        features, neighbors = case.preprocess_eval(
            (current_pos, particle_type), neighbors
        )
        return current_pos, neighbors, features

    return push_forward


@partial(jax.jit, static_argnames=["model_fn", "loss_weight"])
def mse(
    params: hk.Params,
    state: hk.State,
    features: Dict[str, jnp.ndarray],
    particle_type: jnp.ndarray,
    target: jnp.ndarray,
    model_fn: Callable,
    loss_weight: LossWeights,
):
    pred, state = model_fn(params, state, (features, particle_type))
    # check active (non zero) output shapes
    keys = list(set(loss_weight.nonzero) & set(pred.keys()))
    assert all(target[k].shape == pred[k].shape for k in keys)
    # particle mask
    non_kinematic_mask = jnp.logical_not(get_kinematic_mask(particle_type))
    num_non_kinematic = non_kinematic_mask.sum()
    # loss components
    losses = []
    for t in keys:
        losses.append((loss_weight[t] * (pred[t] - target[t]) ** 2).sum(axis=-1))
    total_loss = jnp.array(losses).sum(0)
    total_loss = jnp.where(non_kinematic_mask, total_loss, 0)
    total_loss = total_loss.sum() / num_non_kinematic

    return total_loss, state


@partial(jax.jit, static_argnames=["loss_fn", "opt_update"])
def update(
    params: hk.Module,
    state: hk.State,
    features_batch: Tuple[jraph.GraphsTuple, ...],
    target_batch: Tuple[jnp.ndarray, ...],
    particle_type_batch: Tuple[jnp.ndarray, ...],
    opt_state: optax.OptState,
    loss_fn: Callable,
    opt_update: Callable,
) -> Tuple[float, hk.Params, hk.State, optax.OptState]:
    value_and_grad_vmap = vmap(
        jax.value_and_grad(loss_fn, has_aux=True), in_axes=(None, None, 0, 0, 0)
    )
    (loss, state), grads = value_and_grad_vmap(
        params, state, features_batch, particle_type_batch, target_batch
    )

    # aggregate over the first (batch) dimension of each leave element
    grads = jax.tree_map(lambda x: x.sum(axis=0), grads)
    state = jax.tree_map(lambda x: x.sum(axis=0), state)
    loss = jax.tree_map(lambda x: x.mean(axis=0), loss)

    updates, opt_state = opt_update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return loss, new_params, state, opt_state


def train(
    model: hk.TransformedWithState,
    case: CaseSetupFn,
    params: hk.Params,
    state: hk.State,
    neighbors: partition.NeighborList,
    loader_train: DataLoader,
    loader_eval: DataLoader,
    metrics_computer: MetricsComputer,
    args: Namespace,
):
    # checkpointing and logging
    run_prefix = "_".join([args.config.model, args.info.dataset_name, ""])
    i = 0
    while os.path.isdir(os.path.join(args.config.ckp_dir, run_prefix + str(i))):
        i += 1
    args.info.run_name = run_prefix + str(i)

    ckp_dir = os.path.join(args.config.ckp_dir, args.info.run_name)
    os.makedirs(ckp_dir, exist_ok=True)
    os.makedirs(os.path.join(ckp_dir, "best"), exist_ok=True)

    # save config file
    with open(os.path.join(ckp_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(args.config), f)
    with open(os.path.join(ckp_dir, "best", "config.yaml"), "w") as f:
        yaml.dump(vars(args.config), f)

    # wandb doesn't like Namespace objects
    args_dict = copy.copy(args)
    args_dict.config = vars(args.config)
    args_dict.info = vars(args.info)

    if args.config.wandb:
        wandb.init(
            project=args.config.wandb_project,
            entity="segnn-sph",
            name=args.info.run_name,
            config=args_dict,
            save_code=True,
        )

    # learning rate decays from 1e-4 to 1e-6 over 10M steps exponentially
    lr_scheduler = optax.exponential_decay(
        init_value=args.config.lr_start,
        transition_steps=int(5e6),
        decay_rate=args.config.lr_decay_rate,
        end_value=args.config.lr_final,
    )
    opt_init, opt_update = optax.adamw(learning_rate=lr_scheduler, weight_decay=1e-8)
    # Precompile model for evaluation
    model_apply = jax.jit(model.apply)
    # loss and update functions
    loss_weight = (
        LossWeights(**args.config.loss_weight)
        if hasattr(args.config, "loss_weight") and args.config.loss_weight is not None
        else LossWeights(**{})
    )
    loss_fn = partial(mse, model_fn=model_apply, loss_weight=loss_weight)
    update_fn = partial(update, loss_fn=loss_fn, opt_update=opt_update)

    # continue training from checkpoint or initialize optimizer state
    if args.config.model_dir:
        _, _, opt_state, _ = load_haiku(args.config.model_dir)
    else:
        opt_state = opt_init(params)

    preprocess_vmap = jax.vmap(case.preprocess, in_axes=(0, 0, None, 0, None))

    push_forward = push_forward_build(model_apply, case)
    push_forward_vmap = jax.vmap(push_forward, in_axes=(0, 0, 0, 0, None, None))

    # prepare for batch training.
    key = jax.random.PRNGKey(args.config.seed)
    keys = jax.random.split(key, args.config.batch_size)
    neighbors_batch = broadcast_to_batch(neighbors, args.config.batch_size)

    step_digits = len(str(int(args.config.step_max)))
    step = args.info.step_start
    while step < args.config.step_max:
        for raw_batch in loader_train:
            # numpy to jax
            raw_batch = jax.tree_map(lambda x: jnp.array(x), raw_batch)

            key, unroll_steps = push_forward_sample_steps(
                key, step, args.config.pushforward
            )
            # target computation incorporates the sampled number pushforward steps
            keys, features_batch, target_batch, neighbors_batch = preprocess_vmap(
                keys, raw_batch, args.config.noise_std, neighbors_batch, unroll_steps
            )
            # unroll for push-forward steps
            _current_pos = raw_batch[0][:, :, : args.config.input_seq_length]
            for _ in range(unroll_steps):
                _current_pos, neighbors_batch, features_batch = push_forward_vmap(
                    features_batch,
                    _current_pos,
                    raw_batch[1],
                    neighbors_batch,
                    params,
                    state,
                )

            if neighbors_batch.did_buffer_overflow.sum() > 0:
                # check if the neighbor list is too small for any of the samples
                # if so, reallocate the neighbor list
                ind = jnp.argmax(neighbors_batch.did_buffer_overflow)
                edges_ = neighbors_batch.idx[ind].shape
                print(f"Reallocate neighbors list {edges_} at step {step}")
                sample = broadcast_from_batch(raw_batch, index=ind)
                _, _, _, nbrs = case.allocate(keys[0], sample)
                print(f"To list {nbrs.idx.shape}")

                neighbors_batch = broadcast_to_batch(nbrs, args.config.batch_size)

                # To run the loop N times even if sometimes
                # did_buffer_overflow > 0 we directly return to the beginning
                continue

            loss, params, state, opt_state = update_fn(
                params=params,
                state=state,
                features_batch=features_batch,
                target_batch=target_batch,
                particle_type_batch=raw_batch[1],
                opt_state=opt_state,
            )

            if step % args.config.log_steps == 0:
                if args.config.wandb:
                    wandb.log({"train/loss": loss.item()}, step)
                else:
                    step_str = str(step).zfill(step_digits)
                    print(f"{step_str}, train/loss: {loss.item():.5f}.")

            if step % args.config.eval_steps == 0 and step > 0:
                nbrs = broadcast_from_batch(neighbors_batch, index=0)
                eval_metrics, nbrs = eval_rollout(
                    case=case,
                    metrics_computer=metrics_computer,
                    model_apply=model_apply,
                    params=params,
                    state=state,
                    neighbors=nbrs,
                    loader_eval=loader_eval,
                    n_rollout_steps=args.config.n_rollout_steps,
                    n_trajs=args.config.eval_n_trajs,
                    rollout_dir=args.config.rollout_dir,
                    out_type=args.config.out_type,
                )

                metrics = averaged_metrics(eval_metrics)
                metadata_ckp = {
                    "step": step,
                    "loss": metrics["val/loss"],
                }
                save_haiku(ckp_dir, params, state, opt_state, metadata_ckp)

                if args.config.wandb:
                    wandb.log(metrics, step)
                else:
                    print(metrics)

            step += 1
            if step == args.config.step_max:
                break
