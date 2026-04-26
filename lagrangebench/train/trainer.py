"""Training utils and functions."""

import os
from collections import namedtuple
from functools import partial
from typing import Callable, Dict, Optional, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import wandb
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
    load_model,
    save_model,
    set_seed,
)

from .strats import push_forward_build, push_forward_sample_steps


def _sample_loss(
    model: eqx.Module,
    features: Dict[str, jnp.ndarray],
    particle_type: jnp.ndarray,
    target: Dict[str, jnp.ndarray],
    loss_weight,
) -> jnp.ndarray:
    """Mean-squared error over active (non-kinematic) particles of a single sample."""
    pred = model((features, particle_type))
    assert all(target[k].shape == pred[k].shape for k in pred)

    non_kinematic_mask = jnp.logical_not(get_kinematic_mask(particle_type))
    num_non_kinematic = non_kinematic_mask.sum()

    losses = []
    for t in pred:
        w = getattr(loss_weight, t)
        losses.append((w * (pred[t] - target[t]) ** 2).sum(axis=-1))
    total_loss = jnp.array(losses).sum(0)
    total_loss = jnp.where(non_kinematic_mask, total_loss, 0)
    return total_loss.sum() / num_non_kinematic


def _make_update_fn(opt_update, loss_weight):
    """Build a jitted update step that does batched loss + grad + optim step."""

    def batch_loss(model, features_batch, particle_type_batch, target_batch):
        losses = jax.vmap(_sample_loss, in_axes=(None, 0, 0, 0, None))(
            model, features_batch, particle_type_batch, target_batch, loss_weight
        )
        return losses.mean()

    @eqx.filter_jit
    def update_fn(model, features_batch, target_batch, particle_type_batch, opt_state):
        loss, grads = eqx.filter_value_and_grad(batch_loss)(
            model, features_batch, particle_type_batch, target_batch
        )
        updates, opt_state = opt_update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    return update_fn


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
        model: eqx.Module,
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
            model: Equinox model (already initialized with a PRNG key).
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
        self.cfg_train = OmegaConf.merge(defaults.train, cfg_train)
        self.cfg_eval = OmegaConf.merge(defaults.eval, cfg_eval)
        self.cfg_logging = OmegaConf.merge(defaults.logging, cfg_logging)

        assert isinstance(model, eqx.Module), (
            "Model must be an equinox.Module instance."
        )

        available_rollout_length = data_valid.subseq_length - input_seq_length
        assert self.cfg_eval.n_rollout_steps <= available_rollout_length, (
            "The loss cannot be evaluated on longer than a ground truth trajectory "
            f"({self.cfg_eval.n_rollout_steps} > {available_rollout_length})"
        )
        assert self.cfg_eval.train.n_trajs <= data_valid.num_samples, (
            f"Number of requested validation trajectories exceeds the available ones "
            f"({self.cfg_eval.train.n_trajs} > {data_valid.num_samples})"
        )

        if self.cfg_eval.train.n_trajs == -1:
            self.cfg_eval.train.n_trajs = data_valid.num_samples

        loss_weight = self.cfg_train.loss_weight
        self.loss_weight = namedtuple("loss_weight", loss_weight)(**loss_weight)

        self.base_key, seed_worker, generator = set_seed(seed)

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

        lr_scheduler = optax.exponential_decay(
            init_value=self.cfg_train.optimizer.lr_start,
            transition_steps=self.cfg_train.optimizer.lr_decay_steps,
            decay_rate=self.cfg_train.optimizer.lr_decay_rate,
            end_value=self.cfg_train.optimizer.lr_final,
        )
        self.optimizer = optax.adamw(
            learning_rate=lr_scheduler, weight_decay=1e-8
        )

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
        model: Optional[eqx.Module] = None,
        opt_state: Optional[optax.OptState] = None,
        store_ckp: Optional[str] = None,
        load_ckp: Optional[str] = None,
        wandb_config: Optional[Dict] = None,
    ) -> Tuple[eqx.Module, optax.OptState]:
        """
        Training loop.

        Trains and evals the model on the given case and dataset, and saves the model
        checkpoints and best models.

        Args:
            step_max: Maximum number of training steps.
            model: Optional model. If provided, training continues from it.
            opt_state: Optional optimizer state.
            store_ckp: Checkpoints destination. Without it params aren't saved.
            load_ckp: Initial checkpoint directory. If provided resumes training.
            wandb_config: Optional configuration to be logged on wandb.

        Returns:
            Tuple containing the final trained model and optimizer state.
        """
        case = self.case
        cfg_train = self.cfg_train
        cfg_eval = self.cfg_eval
        cfg_logging = self.cfg_logging
        loader_train = self.loader_train
        loader_valid = self.loader_valid
        noise_std = cfg_train.noise_std
        pushforward = cfg_train.pushforward

        update_fn = _make_update_fn(self.optimizer.update, self.loss_weight)

        if model is None:
            model = self.model

        raw_batch = next(iter(loader_train))
        raw_batch = jax.tree.map(lambda x: jnp.array(x), raw_batch)
        pos_input_and_target, particle_type = raw_batch
        raw_sample = (pos_input_and_target[0], particle_type[0])
        key, features, _, neighbors = case.allocate(self.base_key, raw_sample)

        step = 0
        if load_ckp is not None:
            model, opt_state, step = load_model(load_ckp, model)

        if cfg_logging.wandb:
            if wandb_config is None:
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
                "num_params": int(get_num_params(model)),
                "step_start": step,
            }

            wandb_run = wandb.init(
                project=cfg_logging.wandb_project,
                entity=cfg_logging.wandb_entity,
                name=cfg_logging.run_name,
                config=wandb_config,
                save_code=True,
            )

        if opt_state is None:
            opt_state = self.optimizer.init(eqx.filter(model, eqx.is_inexact_array))

        if store_ckp is not None:
            os.makedirs(store_ckp, exist_ok=True)
            os.makedirs(os.path.join(store_ckp, "best"), exist_ok=True)

        preprocess_vmap = jax.vmap(case.preprocess, in_axes=(0, 0, None, 0, None))
        push_forward = push_forward_build(case)
        push_forward_vmap = eqx.filter_vmap(
            push_forward, in_axes=(None, 0, 0, 0, 0)
        )

        keys = jax.random.split(key, loader_train.batch_size)
        neighbors_batch = broadcast_to_batch(neighbors, loader_train.batch_size)

        while step < step_max + 1:
            for raw_batch in loader_train:
                raw_batch = jax.tree.map(lambda x: jnp.array(x), raw_batch)

                key, unroll_steps = push_forward_sample_steps(key, step, pushforward)
                _keys, features_batch, target_batch, neighbors_batch = preprocess_vmap(
                    keys,
                    raw_batch,
                    noise_std,
                    neighbors_batch,
                    unroll_steps,
                )
                _current_pos = raw_batch[0][:, :, : self.input_seq_length]
                for _ in range(unroll_steps):
                    if neighbors_batch.did_buffer_overflow.sum() > 0:
                        break
                    _current_pos, neighbors_batch, features_batch = push_forward_vmap(
                        model,
                        features_batch,
                        _current_pos,
                        raw_batch[1],
                        neighbors_batch,
                    )

                if neighbors_batch.did_buffer_overflow.sum() > 0:
                    print(f"Reallocate neighbors list at step {step}")
                    ind = jnp.argmax(neighbors_batch.did_buffer_overflow)
                    sample = broadcast_from_batch(raw_batch, index=ind)
                    _, _, _, nbrs = case.allocate(keys[ind], sample, noise_std)
                    print(f"From {neighbors_batch.idx[ind].shape} to {nbrs.idx.shape}")
                    neighbors_batch = broadcast_to_batch(nbrs, loader_train.batch_size)
                    continue
                keys = _keys

                loss, model, opt_state = update_fn(
                    model,
                    features_batch,
                    target_batch,
                    raw_batch[1],
                    opt_state,
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
                        model=model,
                        case=case,
                        metrics_computer=self.metrics_computer,
                        neighbors=nbrs,
                        loader_eval=loader_valid,
                        n_rollout_steps=cfg_eval.n_rollout_steps,
                        n_trajs=cfg_eval.train.n_trajs,
                        rollout_dir=cfg_eval.rollout_dir,
                        out_type=cfg_eval.train.out_type,
                    )
                    metrics = averaged_metrics(eval_metrics)
                    # pick a rollout metric as the "loss" used for best-ckpt
                    # selection (val/loss isn't emitted by MetricsComputer)
                    best_key = next(
                        (
                            k
                            for k in ("val/mse20", "val/mse10", "val/mse5", "val/mse")
                            if k in metrics
                        ),
                        None,
                    )
                    metadata_ckp = {
                        "step": step,
                        "loss": float(metrics[best_key]) if best_key else None,
                    }
                    if store_ckp is not None:
                        save_model(store_ckp, model, opt_state, metadata_ckp)

                    if cfg_logging.wandb:
                        wandb_run.log(metrics, step)
                    else:
                        print(metrics)

                step += 1
                if step == step_max + 1:
                    break

        if cfg_logging.wandb:
            wandb_run.finish()

        return model, opt_state
