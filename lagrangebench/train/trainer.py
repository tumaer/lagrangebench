"""Training utils and functions."""

import os
from collections import namedtuple
from functools import partial
from typing import Callable, Dict, Optional, Tuple, Union

import flax.linen as flax_nn
import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import optax
import wandb
from jax import vmap
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from lagrangebench.data import H5Dataset
from lagrangebench.data.utils import numpy_collate
from lagrangebench.defaults import defaults
from lagrangebench.evaluate import MetricsComputer, averaged_metrics, eval_rollout
from lagrangebench.utils import (
    broadcast_from_batch,
    broadcast_to_batch,
    get_kinematic_mask,
    get_num_params,
    load_haiku,
    save_haiku,
    set_seed,
)

from .strats import push_forward_build, push_forward_sample_steps

ModelType = Union[hk.TransformedWithState, flax_nn.Module]
ParamsType = hk.Params
StateType = hk.State


@partial(jax.jit, static_argnames=["model_fn", "loss_weight"])
def _mse(
    params: ParamsType,
    state: StateType,
    features: Dict[str, jnp.ndarray],
    particle_type: jnp.ndarray,
    target: jnp.ndarray,
    model_fn: Callable,
    loss_weight: Dict[str, float],
):
    if not hasattr(model_fn._fun, "__self__"):
        pred, state = model_fn(params, state, (features, particle_type))
    elif isinstance(model_fn._fun.__self__, flax_nn.Module):
        kwargs = {}
        if state:
            kwargs["mutables"] = state.keys()
        pred = model_fn(
            {"params": params, "batch_stats": state},
            (features, particle_type),
            **kwargs,
        )
        if state:
            pred, updates = pred
            state = updates["batch_stats"]

    # check active (non zero) output shapes
    assert all(target[k].shape == pred[k].shape for k in pred)
    # particle mask
    non_kinematic_mask = jnp.logical_not(get_kinematic_mask(particle_type))
    num_non_kinematic = non_kinematic_mask.sum()
    # loss components
    losses = []
    for t in pred:
        w = getattr(loss_weight, t)
        losses.append((w * (pred[t] - target[t]) ** 2).sum(axis=-1))
    total_loss = jnp.array(losses).sum(0)
    total_loss = jnp.where(non_kinematic_mask, total_loss, 0)
    total_loss = total_loss.sum() / num_non_kinematic

    return total_loss, state


@partial(jax.jit, static_argnames=["loss_fn", "opt_update"])
def _update(
    params: ParamsType,
    state: StateType,
    features_batch: Tuple[jraph.GraphsTuple, ...],
    target_batch: Tuple[jnp.ndarray, ...],
    particle_type_batch: Tuple[jnp.ndarray, ...],
    opt_state: optax.OptState,
    loss_fn: Callable,
    opt_update: Callable,
) -> Tuple[float, ParamsType, StateType, optax.OptState]:
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


class Trainer:
    """
    Trainer class.

    Given a model, case setup, training and validation datasets this class
    automates training and evaluation.

    1. Initializes (or restarts a checkpoint) model, optimizer and loss function.
    2. Trains the model on data_train, using the given pushforward and noise tricks.
    3. Evaluates the model on data_valid on the specified metrics.
    """

    def __init__(
        self,
        model: ModelType,
        case,
        data_train: H5Dataset,
        data_valid: H5Dataset,
        cfg_train: Union[Dict, DictConfig] = defaults.train,
        cfg_eval: Union[Dict, DictConfig] = defaults.eval,
        cfg_logging: Union[Dict, DictConfig] = defaults.logging,
        input_seq_length: int = defaults.model.input_seq_length,
        seed: int = defaults.seed,
    ):
        """Initializes the trainer.

        Args:
            model: (Transformed) Haiku model.
            case: Case setup class.
            data_train: Training dataset.
            data_valid: Validation dataset.
            cfg_train: Training configuration.
            cfg_eval: Evaluation configuration.
            cfg_logging: Logging configuration.
            input_seq_length: Input sequence length, i.e. number of past positions.
            seed: Random seed for model init, training tricks and dataloading.
        """

        if isinstance(cfg_train, Dict):
            cfg_train = OmegaConf.create(cfg_train)
        if isinstance(cfg_eval, Dict):
            cfg_eval = OmegaConf.create(cfg_eval)
        if isinstance(cfg_logging, Dict):
            cfg_logging = OmegaConf.create(cfg_logging)

        self.model = model
        self.case = case
        self.input_seq_length = input_seq_length
        # if one of the cfg_* arguments has a subset of the default configs, merge them
        self.cfg_train = OmegaConf.merge(defaults.train, cfg_train)
        self.cfg_eval = OmegaConf.merge(defaults.eval, cfg_eval)
        self.cfg_logging = OmegaConf.merge(defaults.logging, cfg_logging)

        assert isinstance(
            model, ModelType
        ), "Model must be passed as a Flax model or an Haiku transformed function."

        available_rollout_length = data_valid.subseq_length - input_seq_length
        assert cfg_eval.n_rollout_steps <= available_rollout_length, (
            "The loss cannot be evaluated on longer than a ground truth trajectory "
            f"({cfg_eval.n_rollout_steps} > {available_rollout_length})"
        )
        assert cfg_eval.train.n_trajs <= data_valid.num_samples, (
            f"Number of requested validation trajectories exceeds the available ones "
            f"({cfg_eval.train.n_trajs} > {data_valid.num_samples})"
        )

        # set the number of validation trajectories during training
        if self.cfg_eval.train.n_trajs == -1:
            self.cfg_eval.train.n_trajs = data_valid.num_samples

        # make immutable for jitting
        loss_weight = self.cfg_train.loss_weight
        self.loss_weight = namedtuple("loss_weight", loss_weight)(**loss_weight)

        self.base_key, seed_worker, generator = set_seed(seed)

        # dataloaders
        self.loader_train = DataLoader(
            dataset=data_train,
            batch_size=self.cfg_eval.train.batch_size,
            shuffle=True,
            num_workers=self.cfg_train.num_workers,
            collate_fn=numpy_collate,
            drop_last=True,
            worker_init_fn=seed_worker,
            generator=generator,
        )
        self.loader_valid = DataLoader(
            dataset=data_valid,
            batch_size=self.cfg_eval.infer.batch_size,
            collate_fn=numpy_collate,
            worker_init_fn=seed_worker,
            generator=generator,
        )

        # exponential learning rate decays from lr_start to lr_final over lr_decay_steps
        lr_scheduler = optax.exponential_decay(
            init_value=self.cfg_train.optimizer.lr_start,
            transition_steps=self.cfg_train.optimizer.lr_decay_steps,
            decay_rate=self.cfg_train.optimizer.lr_decay_rate,
            end_value=self.cfg_train.optimizer.lr_final,
        )
        # optimizer
        self.opt_init, self.opt_update = optax.adamw(
            learning_rate=lr_scheduler, weight_decay=1e-8
        )

        # metrics computer config
        self.metrics_computer = MetricsComputer(
            self.cfg_eval.train.metrics,
            dist_fn=self.case.displacement,
            metadata=data_train.metadata,
            input_seq_length=self.input_seq_length,
            stride=self.cfg_eval.train.metrics_stride,
        )

    def train(
        self,
        step_max: int = defaults.train.step_max,
        params: Optional[ParamsType] = None,
        state: Optional[StateType] = None,
        opt_state: Optional[optax.OptState] = None,
        store_ckp: Optional[str] = None,
        load_ckp: Optional[str] = None,
        wandb_config: Optional[Dict] = None,
    ) -> Tuple[ParamsType, StateType, optax.OptState]:
        """
        Training loop.

        Trains and evals the model on the given case and dataset, and saves the model
        checkpoints and best models.

        Args:
            step_max: Maximum number of training steps.
            params: Optional model parameters. If provided, training continues from it.
            state: Optional model state.
            opt_state: Optional optimizer state.
            store_ckp: Checkpoints destination. Without it params aren't saved.
            load_ckp: Initial checkpoint directory. If provided resumes training.
            wandb_config: Optional configuration to be logged on wandb.

        Returns:
            Tuple containing the final model parameters, state and optimizer state.
        """

        model = self.model
        case = self.case
        cfg_train = self.cfg_train
        cfg_eval = self.cfg_eval
        cfg_logging = self.cfg_logging
        loader_train = self.loader_train
        loader_valid = self.loader_valid
        noise_std = cfg_train.noise_std
        pushforward = cfg_train.pushforward

        # Precompile model for evaluation
        model_apply = jax.jit(model.apply)

        # loss and update functions
        loss_fn = partial(_mse, model_fn=model_apply, loss_weight=self.loss_weight)
        update_fn = partial(_update, loss_fn=loss_fn, opt_update=self.opt_update)

        # init values
        pos_input_and_target, particle_type = next(iter(loader_train))
        raw_sample = (pos_input_and_target[0], particle_type[0])
        key, features, _, neighbors = case.allocate(self.base_key, raw_sample)

        step = 0
        if params is not None:
            # continue training from params
            if state is None:
                state = {}
        elif load_ckp:
            # continue training from checkpoint
            # TODO model loading with flax using orbax https://flax.readthedocs.io/en/latest/guides/training_techniques/use_checkpointing.html
            if isinstance(model, flax_nn.Module):
                pass
            else:
                params, state, opt_state, step = load_haiku(load_ckp)
        else:
            # initialize new model
            key, subkey = jax.random.split(key, 2)
            vars = model.init(subkey, (features, particle_type[0]))
            if isinstance(model, flax_nn.Module):
                params = vars["params"]
                state = vars.get("batch_stats", {})
            else:
                params, state = vars

        # start logging
        if cfg_logging.wandb:
            if wandb_config is None:
                # minimal config reconstruction without model details
                wandb_config = {
                    "train": OmegaConf.to_container(cfg_train),
                    "eval": OmegaConf.to_container(cfg_eval),
                    "logging": OmegaConf.to_container(cfg_logging),
                    "dataset_path": loader_train.dataset.dataset_path,
                }

            else:
                wandb_config["eval"]["train"]["n_trajs"] = cfg_eval.train.n_trajs

            wandb_config["info"] = {
                "dataset_name": loader_train.dataset.name,
                "len_train": len(loader_train.dataset),
                "len_eval": len(loader_valid.dataset),
                "num_params": get_num_params(params).item(),
                "step_start": step,
            }

            wandb_run = wandb.init(
                project=cfg_logging.wandb_project,
                entity=cfg_logging.wandb_entity,
                name=cfg_logging.run_name,
                config=wandb_config,
                save_code=True,
            )

        # initialize optimizer state
        if opt_state is None:
            opt_state = self.opt_init(params)

        # create new checkpoint directory
        if store_ckp is not None:
            os.makedirs(store_ckp, exist_ok=True)
            os.makedirs(os.path.join(store_ckp, "best"), exist_ok=True)

        preprocess_vmap = jax.vmap(case.preprocess, in_axes=(0, 0, None, 0, None))
        push_forward = push_forward_build(model_apply, case)
        push_forward_vmap = jax.vmap(push_forward, in_axes=(0, 0, 0, 0, None, None))

        # prepare for batch training.
        keys = jax.random.split(key, loader_train.batch_size)
        neighbors_batch = broadcast_to_batch(neighbors, loader_train.batch_size)

        # start training
        while step < step_max + 1:
            for raw_batch in loader_train:
                # numpy to jax
                raw_batch = jax.tree_map(lambda x: jnp.array(x), raw_batch)

                key, unroll_steps = push_forward_sample_steps(key, step, pushforward)
                # target computation incorporates the sampled number pushforward steps
                _keys, features_batch, target_batch, neighbors_batch = preprocess_vmap(
                    keys,
                    raw_batch,
                    noise_std,
                    neighbors_batch,
                    unroll_steps,
                )
                # unroll for push-forward steps
                _current_pos = raw_batch[0][:, :, : self.input_seq_length]
                for _ in range(unroll_steps):
                    if neighbors_batch.did_buffer_overflow.sum() > 0:
                        break
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

                    print(f"Reallocate neighbors list at step {step}")
                    ind = jnp.argmax(neighbors_batch.did_buffer_overflow)
                    sample = broadcast_from_batch(raw_batch, index=ind)

                    _, _, _, nbrs = case.allocate(keys[ind], sample, noise_std)
                    print(f"From {neighbors_batch.idx[ind].shape} to {nbrs.idx.shape}")
                    neighbors_batch = broadcast_to_batch(nbrs, loader_train.batch_size)

                    # To run the loop N times even if sometimes
                    # did_buffer_overflow > 0 we directly return to the beginning
                    continue
                keys = _keys

                loss, params, state, opt_state = update_fn(
                    params=params,
                    state=state,
                    features_batch=features_batch,
                    target_batch=target_batch,
                    particle_type_batch=raw_batch[1],
                    opt_state=opt_state,
                )

                if step % cfg_logging.log_steps == 0:
                    loss.block_until_ready()
                    if cfg_logging.wandb:
                        wandb_run.log({"train/loss": loss.item()}, step)
                    else:
                        step_str = str(step).zfill(len(str(int(step_max))))
                        print(f"{step_str}, train/loss: {loss.item():.5f}.")

                if step % cfg_logging.eval_steps == 0 and step > 0:
                    nbrs = broadcast_from_batch(neighbors_batch, index=0)
                    eval_metrics = eval_rollout(
                        case=case,
                        metrics_computer=self.metrics_computer,
                        model_apply=model_apply,
                        params=params,
                        state=state,
                        neighbors=nbrs,
                        loader_eval=loader_valid,
                        n_rollout_steps=cfg_eval.n_rollout_steps,
                        n_trajs=cfg_eval.train.n_trajs,
                        rollout_dir=cfg_eval.rollout_dir,
                        out_type=cfg_eval.train.out_type,
                    )

                    metrics = averaged_metrics(eval_metrics)
                    metadata_ckp = {
                        "step": step,
                        "loss": metrics.get("val/loss", None),
                    }
                    if store_ckp is not None:
                        save_haiku(store_ckp, params, state, opt_state, metadata_ckp)

                    if cfg_logging.wandb:
                        wandb_run.log(metrics, step)
                    else:
                        print(metrics)

                step += 1
                if step == step_max + 1:
                    break

        if cfg_logging.wandb:
            wandb_run.finish()

        return params, state, opt_state
