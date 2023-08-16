"""Trainer method."""

import os
from functools import partial
from typing import Callable, Dict, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import optax
from jax import vmap
from torch.utils.data import DataLoader
from wandb.wandb_run import Run

from lagrangebench.data import H5Dataset
from lagrangebench.data.utils import numpy_collate
from lagrangebench.defaults import defaults
from lagrangebench.evaluate import MetricsComputer, averaged_metrics, eval_rollout
from lagrangebench.utils import (
    LossConfig,
    PushforwardConfig,
    broadcast_from_batch,
    broadcast_to_batch,
    get_kinematic_mask,
    get_num_params,
    load_haiku,
    save_haiku,
    set_seed,
)

from .strats import push_forward_build, push_forward_sample_steps


@partial(jax.jit, static_argnames=["model_fn", "loss_weight"])
def _mse(
    params: hk.Params,
    state: hk.State,
    features: Dict[str, jnp.ndarray],
    particle_type: jnp.ndarray,
    target: jnp.ndarray,
    model_fn: Callable,
    loss_weight: LossConfig,
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
def _update(
    params: hk.Params,
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


def Trainer(
    model: hk.TransformedWithState,
    case,
    dataset_train: H5Dataset,
    dataset_eval: H5Dataset,
    lr_start: float = defaults.lr_start,
    batch_size: int = defaults.batch_size,
    input_seq_length: int = defaults.input_seq_length,
    pushforward: Optional[PushforwardConfig] = None,
    noise_std: float = defaults.noise_std,
    metrics: Optional[Dict] = None,
    seed: int = defaults.seed,
    **kwargs,
) -> Callable:
    """
    Trainer builder function.

    Returns a function that trains the model on the given
    case and dataset_train, evaluating it on dataset_eval with the specified metrics.

    Args:
        model: (Transformed) Haiku model.
        case: Case setup class.
        dataset_train: Training dataset.
        dataset_eval: Validation dataset.
        lr_start: Initial learning rate.
        batch_size: Training batch size.
        input_seq_length: Input sequence length. Default is 6.
        pushforward: Pushforward configuration.
        noise_std: Noise standard deviation for the GNS-style noise.
        metrics: Metrics to evaluate the model on.
        seed: Random seed.

    Keyword Args:
        lr_final: Final learning rate.
        lr_decay_steps: Number of steps to reach the final learning rate.
        lr_decay_rate: Learning rate decay rate.
        loss_weight: Loss weight object.
        n_rollout_steps: Number of autoregressive rollout steps.
        eval_n_trajs: Number of trajectories to evaluate.
        rollout_dir: Rollout directory.
        out_type: Output type.
        log_steps: Wandb/screen logging frequency.
        eval_steps: Evaluation and checkpointing frequency.


    Returns:
        Configured training function.
    """
    assert isinstance(
        model, hk.TransformedWithState
    ), "Model must be passed as an Haiku transformed function."

    base_key, seed_worker, generator = set_seed(seed)

    # dataloaders
    loader_train = DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=numpy_collate,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    loader_eval = DataLoader(
        dataset=dataset_eval,
        batch_size=1,
        collate_fn=numpy_collate,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    # learning rate decays from lr_start to lr_final over lr_decay_steps exponentially
    lr_scheduler = optax.exponential_decay(
        init_value=lr_start,
        transition_steps=kwargs.get("lr_decay_steps", defaults.lr_decay_steps),
        decay_rate=kwargs.get("lr_decay_rate", defaults.lr_decay_rate),
        end_value=kwargs.get("lr_final", defaults.lr_final),
    )
    # optimizer
    opt_init, opt_update = optax.adamw(learning_rate=lr_scheduler, weight_decay=1e-8)

    # loss config
    loss_weight = kwargs.get("loss_weight", None)
    if loss_weight is None:
        loss_weight = LossConfig()
    else:
        loss_weight = LossConfig(**loss_weight)
    # pushforward config
    pushforward = kwargs.get("pushforward", None)
    if pushforward is None:
        pushforward = PushforwardConfig()

    # trajectory and rollout
    n_rollout_steps = kwargs.get("n_rollout_steps", defaults.n_rollout_steps)
    eval_n_trajs = kwargs.get("eval_n_trajs", defaults.eval_n_trajs)
    rollout_dir = kwargs.get("rollout_dir", defaults.rollout_dir)
    out_type = kwargs.get("out_type", defaults.out_type)
    # logging and checkpointing
    log_steps = kwargs.get("log_steps", defaults.log_steps)
    eval_steps = kwargs.get("eval_steps", defaults.eval_steps)

    # metrics computer config
    metrics_computer = MetricsComputer(
        metrics,
        dist_fn=case.displacement,
        metadata=dataset_train.metadata,
        input_seq_length=dataset_train.input_seq_length,
    )

    def _train(
        step_max: int = defaults.step_max,
        params: Optional[hk.Params] = None,
        state: Optional[hk.State] = None,
        opt_state: Optional[optax.OptState] = None,
        store_checkpoint: Optional[str] = None,
        load_checkpoint: Optional[str] = None,
        wandb_run: Optional[Run] = None,
    ) -> Tuple[hk.Params, hk.State, optax.OptState]:
        """
        Training function.

        Trains and evals the model on the given case and dataset, and saves the model
        checkpoints and best models.

        Args:
            step_max: Maximum number of training steps.
            params: Optional model parameters. If provided, training continues from it.
            state: Optional model state.
            opt_state: Optional optimizer state.
            store_checkpoint: Checkpoints destination. Without it params aren't saved.
            load_checkpoint: Initial checkpoint directory. If provided resumes training.
            wandb_run: Wandb run.

        Returns:
            Tuple containing the final model parameters, state and optimizer state.
        """
        assert n_rollout_steps <= dataset_eval.subsequence_length - input_seq_length, (
            "If you want to evaluate the loss on more than the ground truth trajectory "
            "length, then use the --eval_n_more_steps argument."
        )
        assert eval_n_trajs <= len(
            loader_eval
        ), "eval_n_trajs must be <= len(loader_valid)"

        # Precompile model for evaluation
        model_apply = jax.jit(model.apply)

        # loss and update functions
        loss_fn = partial(_mse, model_fn=model_apply, loss_weight=loss_weight)
        update_fn = partial(_update, loss_fn=loss_fn, opt_update=opt_update)

        # init values
        pos_input_and_target, particle_type = next(iter(loader_train))
        raw_sample = (pos_input_and_target[0], particle_type[0])
        key, features, _, neighbors = case.allocate(base_key, raw_sample)

        step = 0
        if params is not None:
            # continue training from params
            if state is None:
                state = {}
        elif load_checkpoint:
            # continue training from checkpoint
            params, state, opt_state, step = load_haiku(load_checkpoint)
        else:
            # initialize new model
            key, subkey = jax.random.split(key, 2)
            params, state = model.init(subkey, (features, particle_type[0]))

        if wandb_run is not None:
            wandb_run.log({"info/num_params": get_num_params(params)})
            wandb_run.log({"info/step_start": step})

        # initialize optimizer state
        if opt_state is None:
            opt_state = opt_init(params)

        # create new checkpoint directory
        if store_checkpoint is not None:
            os.makedirs(store_checkpoint, exist_ok=True)
            os.makedirs(os.path.join(store_checkpoint, "best"), exist_ok=True)

        preprocess_vmap = jax.vmap(case.preprocess, in_axes=(0, 0, None, 0, None))
        push_forward = push_forward_build(model_apply, case)
        push_forward_vmap = jax.vmap(push_forward, in_axes=(0, 0, 0, 0, None, None))

        # prepare for batch training.
        keys = jax.random.split(key, loader_train.batch_size)
        neighbors_batch = broadcast_to_batch(neighbors, loader_train.batch_size)

        # start training
        while step < step_max:
            for raw_batch in loader_train:
                # numpy to jax
                raw_batch = jax.tree_map(lambda x: jnp.array(x), raw_batch)

                key, unroll_steps = push_forward_sample_steps(key, step, pushforward)
                # target computation incorporates the sampled number pushforward steps
                keys, features_batch, target_batch, neighbors_batch = preprocess_vmap(
                    keys,
                    raw_batch,
                    noise_std,
                    neighbors_batch,
                    unroll_steps,
                )
                # unroll for push-forward steps
                _current_pos = raw_batch[0][:, :, :input_seq_length]
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

                    neighbors_batch = broadcast_to_batch(nbrs, loader_train.batch_size)

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

                if step % log_steps == 0:
                    loss.block_until_ready()
                    if wandb_run:
                        wandb_run.log({"train/loss": loss.item()}, step)
                    else:
                        step_str = str(step).zfill(len(str(int(step_max))))
                        print(f"{step_str}, train/loss: {loss.item():.5f}.")

                if step % eval_steps == 0 and step > 0:
                    nbrs = broadcast_from_batch(neighbors_batch, index=0)
                    eval_metrics = eval_rollout(
                        case=case,
                        metrics_computer=metrics_computer,
                        model_apply=model_apply,
                        params=params,
                        state=state,
                        neighbors=nbrs,
                        loader_eval=loader_eval,
                        n_rollout_steps=n_rollout_steps,
                        n_trajs=eval_n_trajs,
                        rollout_dir=rollout_dir,
                        out_type=out_type,
                    )

                    metrics = averaged_metrics(eval_metrics)
                    metadata_ckp = {
                        "step": step,
                        "loss": metrics.get("val/loss", None),
                    }
                    if store_checkpoint is not None:
                        save_haiku(
                            store_checkpoint, params, state, opt_state, metadata_ckp
                        )

                    if wandb_run:
                        wandb_run.log(metrics, step)
                    else:
                        print(metrics)

                step += 1
                if step == step_max:
                    break

        return params, state, opt_state

    return _train
