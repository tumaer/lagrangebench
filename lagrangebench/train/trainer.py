"""Training utils and functions."""

import os
import time
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import optax
from jax import random, vmap
from torch.utils.data import DataLoader
from wandb.wandb_run import Run

from lagrangebench.data import H5Dataset
from lagrangebench.data.utils import numpy_collate
from lagrangebench.defaults import defaults
from lagrangebench.evaluate import (
    MetricsComputer,
    averaged_metrics,
    eval_rollout,
    eval_rollout_pde_refiner,
)
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


# MSE used to define loss function
@partial(jax.jit, static_argnames=["model_fn", "loss_weight"])  #
def _mse(
    params: hk.Params,
    state: hk.State,  # for GNS, state is an empty dictionary
    features: Dict[str, jnp.ndarray],
    particle_type: jnp.ndarray,
    target: jnp.ndarray,
    model_fn: Callable,  # jitted model function
    loss_weight: LossConfig,  # LossConfig(pos=0.0, vel=0.0, acc=1.0) by default
):
    pred, state = model_fn(
        params, state, (features, particle_type)
    )  # returns the predicted state (could be position or velocity or acceleration(default))
    # for RPF_2D: (3200,2)
    # check active (non zero) output shapes
    keys = list(set(loss_weight.nonzero) & set(pred.keys()))  # keys = ['acc'] for GNS
    assert all(
        target[k].shape == pred[k].shape for k in keys
    )  # asserting if target['acc'] and pred['acc'] have same shape
    # particle mask
    non_kinematic_mask = jnp.logical_not(
        get_kinematic_mask(particle_type)
    )  # kinematic massk is for obsatcles like moving wall or rigid wall,
    # logical not will return 'True' for fluid particles
    num_non_kinematic = (
        non_kinematic_mask.sum()
    )  # returns [3200] for batch size 1; [3200,3200] for batch size 2. in this function, the comments are written wrt. batch size 1
    # loss components
    losses = []
    for t in keys:
        losses.append(
            (loss_weight[t] * (pred[t] - target[t]) ** 2).sum(axis=-1)
        )  # pred['acc'].shape =(3200,2) and target['acc'].shape =(3200,2), sum(axis=-1) will result in shape (3200,)
        # as the last axis (column) is summed, therefore the resulting shape is the rows.                                                             #but when appended to a list, the shape is (1,3200)
    total_loss = jnp.array(losses).sum(
        0
    )  # now after summing across the row, the shape is (3200,)
    total_loss = jnp.where(
        non_kinematic_mask, total_loss, 0
    )  # loss is 0 for 'non-fluid' particles.
    total_loss = (
        total_loss.sum() / num_non_kinematic
    )  # normalizing with the total number of particles

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
    data_train: H5Dataset,
    data_eval: H5Dataset,
    pushforward: Optional[PushforwardConfig] = None,
    metrics: List = ["mse"],
    seed: int = defaults.seed,
    batch_size: int = defaults.batch_size,
    input_seq_length: int = defaults.input_seq_length,
    noise_std: float = defaults.noise_std,
    lr_start: float = defaults.lr_start,
    lr_final: float = defaults.lr_final,
    lr_decay_steps: int = defaults.lr_decay_steps,
    lr_decay_rate: float = defaults.lr_decay_rate,
    loss_weight: Optional[LossConfig] = None,
    n_rollout_steps: int = defaults.n_rollout_steps,
    eval_n_trajs: int = defaults.eval_n_trajs,
    rollout_dir: str = defaults.rollout_dir,
    out_type: str = defaults.out_type,
    log_steps: int = defaults.log_steps,
    eval_steps: int = defaults.eval_steps,
    metrics_stride: int = defaults.metrics_stride,
    is_pde_refiner: bool = defaults.is_pde_refiner,
    num_refinement_steps: int = defaults.num_refinement_steps,
    sigma_min: float = defaults.sigma_min,
    **kwargs,
) -> Callable:
    """
    Builds a function that automates model training and evaluation.

    Given a model, training and validation datasets and a case this function returns
    another function that:

    1. Initializes (or resumes from a checkpoint) model, optimizer and loss function.
    2. Trains the model on data_train, using the given pushforward and noise tricks.
    3. Evaluates the model on data_eval on the specified metrics.

    Args:
        model: (Transformed) Haiku model.
        case: Case setup class.
        data_train: Training dataset.
        data_eval: Validation dataset.
        pushforward: Pushforward configuration. None for no pushforward.
        metrics: Metrics to evaluate the model on.
        seed: Random seed for model init, training tricks and dataloading.
        batch_size: Training batch size.
        input_seq_length: Input sequence length. Default is 6.
        noise_std: Noise standard deviation for the GNS-style noise.
        lr_start: Initial learning rate.
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
        dataset=data_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # to parallelly fetch data from the dataset
        collate_fn=numpy_collate,
        drop_last=True,  # drops the last batch if it is not of batch_size
        worker_init_fn=seed_worker,
        generator=generator,
    )
    loader_eval = DataLoader(
        dataset=data_eval,
        batch_size=1,
        collate_fn=numpy_collate,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    # learning rate decays from lr_start to lr_final over lr_decay_steps exponentially
    lr_scheduler = optax.exponential_decay(
        init_value=lr_start,  # 0.0005
        transition_steps=lr_decay_steps,  # 1,00,000
        decay_rate=lr_decay_rate,  # 0.1
        end_value=lr_final,  # 1e-6
    )
    # optimizer
    opt_init, opt_update = optax.adamw(learning_rate=lr_scheduler, weight_decay=1e-8)

    # loss config (determines on what the loss would be calculated, would it be position OR velocity OR acceleration)
    # it can be set in defaults.yaml
    if loss_weight is None:
        loss_weight = (
            LossConfig()
        )  # the object by default is: LossConfig(pos=0.0, vel=0.0, acc=1.0)
    else:
        loss_weight = LossConfig(**loss_weight)
    # pushforward config
    if pushforward is None:
        pushforward = PushforwardConfig()

    # metrics computer config
    metrics_computer = MetricsComputer(
        metrics,
        dist_fn=case.displacement,
        metadata=data_train.metadata,
        input_seq_length=data_train.input_seq_length,
        stride=metrics_stride,
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
        Training loop.

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
        assert n_rollout_steps <= data_eval.subseq_length - input_seq_length, (
            "You cannot evaluate the loss on longer than a ground truth trajectory "
            f"({n_rollout_steps}, {data_eval.subseq_length}, {input_seq_length})"
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
        pos_input_and_target, particle_type = next(
            iter(loader_train)
        )  # pos_input_and_target.shape =(1,3200,7,2) and particle_type.shape =(1,3200)
        # pos_input_and_target[0].shape =(3200,7,2) and particle_type[0].shape =(3200,)
        raw_sample = (
            pos_input_and_target[0],
            particle_type[0],
        )  # creates an array by concatenating pos_input_and_target[0] and particle_type[0]

        if is_pde_refiner:
            key, subkey = jax.random.split(base_key, 2)
            k = random.randint(subkey, (), 0, num_refinement_steps + 1)
            ##second denoinsing model only
            #k = random.randint(subkey, (), 1, num_refinement_steps + 1)
            is_k_zero = jnp.where(k == 0, True, False)
            key, features, _, neighbors = case.allocate_pde_refiner(
                key, raw_sample, k, is_k_zero, sigma_min, num_refinement_steps
            )
            preprocess_vmap = jax.vmap(
                case.preprocess_pde_refiner,
                in_axes=(0, 0, None, 0, None, None, None, None, None),
            )

        else:
            key, features, _, neighbors = case.allocate(base_key, raw_sample)
            preprocess_vmap = jax.vmap(case.preprocess, in_axes=(0, 0, None, 0, None))

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
            wandb_run.log({"info/num_params": get_num_params(params)}, 0)
            wandb_run.log({"info/step_start": step}, 0)

        # initialize optimizer state
        if opt_state is None:
            opt_state = opt_init(params)

        # create new checkpoint directory
        if store_checkpoint is not None:
            os.makedirs(store_checkpoint, exist_ok=True)
            os.makedirs(os.path.join(store_checkpoint, "best"), exist_ok=True)

        push_forward = push_forward_build(model_apply, case)
        push_forward_vmap = jax.vmap(push_forward, in_axes=(0, 0, 0, 0, None, None))

        # prepare for batch training.
        keys = jax.random.split(
            key, loader_train.batch_size
        )  # by default, batch_size =1
        neighbors_batch = broadcast_to_batch(
            neighbors, loader_train.batch_size
        )  # finds the neighbour list for several batches.
        # For example, if there is a batch of 2, it means we have two different examples which have different initial conditions, each with shape (3200,7,2)

        # print(jax.tree_map(lambda x: x.shape, params)) # print this in the debug console

        # start training and ocassionally evaluate
        while step < step_max + 1:
            for raw_batch in loader_train:
                # numpy to jax  (jax.tree_map converts every item in raw_batch to jnp.array)
                raw_batch = jax.tree_map(
                    lambda x: jnp.array(x), raw_batch
                )  # if batch size is 2, raw_batch.shape =(2,3200,7,2)
                # if batch_size is 1, raw_batch.shape =(1,3200,7,2)
                key, unroll_steps = push_forward_sample_steps(key, step, pushforward)
                # target computation incorporates the sampled number pushforward steps

                if is_pde_refiner:
                    key, subkey = jax.random.split(
                        key, 2
                    )  # upon splitting, both key and subkey are different from the original key which is passed in the argument
                    k = random.randint(subkey, (), 0, num_refinement_steps + 1)
                    #second denoising model as a part of 2 GNNs
                    #k = random.randint(subkey, (), 1, num_refinement_steps + 1)
                    is_k_zero = True if k == 0 else False
                    # in this case, preprocess_vmap() is case.preprocess_pde_refiner
                    (
                        keys,
                        features_batch,
                        target_batch,
                        neighbors_batch,
                    ) = preprocess_vmap(
                        keys,
                        raw_batch,
                        noise_std,
                        neighbors_batch,
                        k,
                        is_k_zero,
                        sigma_min,
                        num_refinement_steps,
                        unroll_steps,
                    )

                else:
                    (
                        keys,
                        features_batch,
                        target_batch,
                        neighbors_batch,
                    ) = preprocess_vmap(
                        keys,
                        raw_batch,
                        noise_std,
                        neighbors_batch,
                        unroll_steps,
                    )

                # unroll for push-forward steps
                _current_pos = raw_batch[0][:, :, :input_seq_length]
                # raw_batch[0].shape =(1,3200,7,2) and _current_pos.shape =(1,3200,6,2)
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

                # Printing training loss
                if step % log_steps == 0:
                    loss.block_until_ready()
                    if wandb_run:
                        wandb_run.log({"train/loss": loss.item()}, step)
                    else:
                        step_str = str(step).zfill(len(str(int(step_max))))
                        print(f"{step_str}, train/loss: {loss.item():.5f}.")

                if (
                    step % eval_steps == 0 and step > 0
                ):  # evaluation(cross validation every 10,000 steps)
                    nbrs = broadcast_from_batch(neighbors_batch, index=0)

                    # key needs to be changed for BATCHED training
                    if is_pde_refiner:
                        eval_metrics = eval_rollout_pde_refiner(
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
                            key=key,
                            num_refinement_steps=num_refinement_steps,
                            sigma_min=sigma_min,
                            out_type=out_type,
                        )

                    else:
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
                if step == step_max + 1:  # step_max= 500000 set in defaults.yaml
                    break

        return params, state, opt_state

    return _train
