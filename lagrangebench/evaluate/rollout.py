"""Evaluation and inference functions for generating rollouts."""

import os
import pickle
import time
from functools import partial
from typing import Callable, Dict, Iterable, Optional, Tuple, Union

import haiku as hk
import jax
import jax.numpy as jnp
import jax_sph.jax_md.partition as partition
from jax import jit, vmap
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from lagrangebench.data import H5Dataset
from lagrangebench.data.utils import numpy_collate
from lagrangebench.defaults import defaults
from lagrangebench.evaluate.metrics import MetricsComputer, MetricsDict
from lagrangebench.evaluate.utils import write_vtk
from lagrangebench.utils import (
    broadcast_from_batch,
    broadcast_to_batch,
    get_kinematic_mask,
    load_haiku,
    set_seed,
)


@partial(jit, static_argnames=["model_apply", "case_integrate"])
def _forward_eval(
    params: hk.Params,
    state: hk.State,
    sample: Tuple[jnp.ndarray, jnp.ndarray],
    current_positions: jnp.ndarray,
    target_positions: jnp.ndarray,
    model_apply: Callable,
    case_integrate: Callable,
) -> jnp.ndarray:
    """Run one update of the 'current_state' using the trained model

    Args:
        params: Haiku model parameters
        state: Haiku model state
        current_positions: Set of historic positions of shape (n_nodel, t_window, dim)
        target_positions: used to get the next state of kinematic particles, i.e. those
            who are not update using the ML model, e.g. boundary particles
        model_apply: model function
        case_integrate: integration function from case.integrate

    Return:
        current_positions: after shifting the historic position sequence by one, i.e. by
            the newly computed most recent position
    """
    _, particle_type = sample

    # predict acceleration and integrate
    pred, state = model_apply(params, state, sample)

    next_position = case_integrate(pred, current_positions)

    # update only the positions of non-boundary particles
    kinematic_mask = get_kinematic_mask(particle_type)
    next_position = jnp.where(
        kinematic_mask[:, None],
        target_positions,
        next_position,
    )

    current_positions = jnp.concatenate(
        [current_positions[:, 1:], next_position[:, None, :]], axis=1
    )  # as next model input

    return current_positions, state


def _eval_batched_rollout(
    forward_eval_vmap: Callable,
    preprocess_eval_vmap: Callable,
    case,
    params: hk.Params,
    state: hk.State,
    traj_batch_i: Tuple[jnp.ndarray, jnp.ndarray],
    neighbors: partition.NeighborList,
    metrics_computer_vmap: Callable,
    n_rollout_steps: int,
    t_window: int,
    n_extrap_steps: int = 0,
) -> Tuple[jnp.ndarray, MetricsDict, jnp.ndarray]:
    """Compute the rollout on a single trajectory.

    Args:
        forward_eval_vmap: Model function.
        case: CaseSetupFn class.
        params: Haiku params.
        state: Haiku state.
        traj_batch_i: Trajectory to evaluate.
        neighbors: Neighbor list.
        metrics_computer: Vectorized MetricsComputer with the desired metrics.
        n_rollout_steps: Number of rollout steps.
        t_window: Length of the input sequence.
        n_extrap_steps: Number of extrapolation steps (beyond the ground truth rollout).

    Returns:
        A tuple with (predicted rollout, metrics, neighbor list).
    """
    # particle type is treated as a static property defined by state at t=0
    pos_input_batch, particle_type_batch = traj_batch_i
    # current_batch_size might be < eval_batch_size if the last batch is not full
    current_batch_size, n_nodes_max, _, dim = pos_input_batch.shape

    # if n_rollout_steps set to -1, use the whole trajectory
    if n_rollout_steps == -1:
        n_rollout_steps = pos_input_batch.shape[2] - t_window

    current_positions_batch = pos_input_batch[:, :, 0:t_window]
    # (batch, n_nodes, t_window, dim)
    traj_len = n_rollout_steps + n_extrap_steps
    target_positions_batch = pos_input_batch[:, :, t_window : t_window + traj_len]

    predictions_batch = jnp.zeros((current_batch_size, traj_len, n_nodes_max, dim))
    neighbors_batch = broadcast_to_batch(neighbors, current_batch_size)

    step = 0
    while step < n_rollout_steps + n_extrap_steps:
        sample_batch = (current_positions_batch, particle_type_batch)

        # 1. preprocess features
        features_batch, neighbors_batch = preprocess_eval_vmap(
            sample_batch, neighbors_batch
        )

        # 2. check whether list overflowed and fix it if so
        if neighbors_batch.did_buffer_overflow.sum() > 0:
            # check if the neighbor list is too small for any of the samples
            # if so, reallocate the neighbor list

            print(f"(eval) Reallocate neighbors list at step {step}")
            ind = jnp.argmax(neighbors_batch.did_buffer_overflow)
            sample = broadcast_from_batch(sample_batch, index=ind)

            _, nbrs_temp = case.allocate_eval(sample)
            print(
                f"(eval) From {neighbors_batch.idx[ind].shape} to {nbrs_temp.idx.shape}"
            )
            neighbors_batch = broadcast_to_batch(nbrs_temp, current_batch_size)

            # To run the loop N times even if sometimes
            # did_buffer_overflow > 0 we directly return to the beginning
            continue

        # 3. run forward model
        current_positions_batch, state_batch = forward_eval_vmap(
            params,
            state,
            (features_batch, particle_type_batch),
            current_positions_batch,
            target_positions_batch[:, :, step],
        )
        # the state is not passed out of this loop, so no not really relevant
        state = broadcast_from_batch(state_batch, 0)

        # 4. write predicted next position to output array
        predictions_batch = predictions_batch.at[:, step].set(
            current_positions_batch[:, :, -1]  # most recently predicted positions
        )

        step += 1

    # (batch, n_nodes, time, dim) -> (batch, time, n_nodes, dim)
    target_positions_batch = target_positions_batch.transpose(0, 2, 1, 3)
    # slice out extrapolation steps
    metrics_batch = metrics_computer_vmap(
        predictions_batch[:, :n_rollout_steps, :, :], target_positions_batch
    )

    return (predictions_batch, metrics_batch, broadcast_from_batch(neighbors_batch, 0))


def eval_rollout(
    model_apply: Callable,
    case,
    params: hk.Params,
    state: hk.State,
    loader_eval: Iterable,
    neighbors: partition.NeighborList,
    metrics_computer: MetricsComputer,
    n_rollout_steps: int,
    n_trajs: int,
    rollout_dir: str,
    out_type: str = "none",
    n_extrap_steps: int = 0,
) -> MetricsDict:
    """Compute the rollout and evaluate the metrics.

    Args:
        model_apply: Model function.
        case: CaseSetupFn class.
        params: Haiku params.
        state: Haiku state.
        loader_eval: Evaluation data loader.
        neighbors: Neighbor list.
        metrics_computer: MetricsComputer with the desired metrics.
        n_rollout_steps: Number of rollout steps.
        n_trajs: Number of ground truth trajectories to evaluate.
        rollout_dir: Parent directory path where to store the rollout and metrics dict.
        out_type: Output type. Either "none", "vtk" or "pkl".
        n_extrap_steps: Number of extrapolation steps (beyond the ground truth rollout).

    Returns:
        Metrics per trajectory.
    """
    batch_size = loader_eval.batch_size
    t_window = loader_eval.dataset.input_seq_length
    eval_metrics = {}

    if rollout_dir is not None:
        os.makedirs(rollout_dir, exist_ok=True)

    forward_eval = partial(
        _forward_eval,
        model_apply=model_apply,
        case_integrate=case.integrate,
    )
    forward_eval_vmap = vmap(forward_eval, in_axes=(None, None, 0, 0, 0))
    preprocess_eval_vmap = vmap(case.preprocess_eval, in_axes=(0, 0))
    metrics_computer_vmap = vmap(metrics_computer, in_axes=(0, 0))

    for i, traj_batch_i in enumerate(loader_eval):
        # if n_trajs is not a multiple of batch_size, we slice from the last batch
        n_traj_left = n_trajs - i * batch_size
        if n_traj_left < batch_size:
            traj_batch_i = jax.tree_map(lambda x: x[:n_traj_left], traj_batch_i)

        # numpy to jax
        traj_batch_i = jax.tree_map(lambda x: jnp.array(x), traj_batch_i)
        # (pos_input_batch, particle_type_batch) = traj_batch_i
        # pos_input_batch.shape = (batch, num_particles, seq_length, dim)

        example_rollout_batch, metrics_batch, neighbors = _eval_batched_rollout(
            forward_eval_vmap=forward_eval_vmap,
            preprocess_eval_vmap=preprocess_eval_vmap,
            case=case,
            params=params,
            state=state,
            traj_batch_i=traj_batch_i,  # (batch, nodes, t, dim)
            neighbors=neighbors,
            metrics_computer_vmap=metrics_computer_vmap,
            n_rollout_steps=n_rollout_steps,
            t_window=t_window,
            n_extrap_steps=n_extrap_steps,
        )

        current_batch_size = traj_batch_i[0].shape[0]
        for j in range(current_batch_size):
            # write metrics to output dictionary
            ind = i * batch_size + j
            eval_metrics[f"rollout_{ind}"] = broadcast_from_batch(metrics_batch, j)

        if rollout_dir is not None:
            # (batch, nodes, t, dim) -> (batch, t, nodes, dim)
            pos_input_batch = traj_batch_i[0].transpose(0, 2, 1, 3)

            for j in range(current_batch_size):  # write every trajectory to file
                pos_input = pos_input_batch[j]
                example_rollout = example_rollout_batch[j]

                initial_positions = pos_input[:t_window]
                example_full = jnp.concatenate([initial_positions, example_rollout])
                example_rollout = {
                    "predicted_rollout": example_full,  # (t + extrap, nodes, dim)
                    "ground_truth_rollout": pos_input,  # (t, nodes, dim),
                    "particle_type": traj_batch_i[1][j],  # (nodes,)
                }

                file_prefix = os.path.join(rollout_dir, f"rollout_{i*batch_size+j}")
                if out_type == "vtk":  # write vtk files for each time step
                    for k in range(example_full.shape[0]):
                        # predictions
                        state_vtk = {
                            "r": example_rollout["predicted_rollout"][k],
                            "tag": example_rollout["particle_type"],
                        }
                        write_vtk(state_vtk, f"{file_prefix}_{k}.vtk")
                    for k in range(pos_input.shape[0]):
                        # ground truth reference
                        ref_state_vtk = {
                            "r": example_rollout["ground_truth_rollout"][k],
                            "tag": example_rollout["particle_type"],
                        }
                        write_vtk(ref_state_vtk, f"{file_prefix}_ref_{k}.vtk")
                elif out_type == "pkl":
                    filename = f"{file_prefix}.pkl"

                    with open(filename, "wb") as f:
                        pickle.dump(example_rollout, f)

        if (i * batch_size + j + 1) >= n_trajs:
            break

    if rollout_dir is not None:
        # save metrics
        t = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        with open(f"{rollout_dir}/metrics{t}.pkl", "wb") as f:
            pickle.dump(eval_metrics, f)

    return eval_metrics


def infer(
    model: hk.TransformedWithState,
    case,
    data_test: H5Dataset,
    params: Optional[hk.Params] = None,
    state: Optional[hk.State] = None,
    load_ckp: Optional[str] = None,
    cfg_eval_infer: Union[Dict, DictConfig] = defaults.eval.infer,
    rollout_dir: Optional[str] = defaults.eval.rollout_dir,
    n_rollout_steps: int = defaults.eval.n_rollout_steps,
    seed: int = defaults.seed,
):
    """
    Infer on a dataset, compute metrics and optionally save rollout in out_type format.

    Args:
        model: (Transformed) Haiku model.
        case: Case setup class.
        data_test: Test dataset.
        params: Haiku params.
        state: Haiku state.
        load_ckp: Path to checkpoint directory.
        rollout_dir: Path to rollout directory.
        cfg_eval_infer: Evaluation configuration for inference mode.
        n_rollout_steps: Number of rollout steps.
        seed: Seed.

    Returns:
        eval_metrics: Metrics per trajectory.
    """
    assert (
        params is not None or load_ckp is not None
    ), "Either params or a load_ckp directory must be provided for inference."

    if isinstance(cfg_eval_infer, Dict):
        cfg_eval_infer = OmegaConf.create(cfg_eval_infer)

    # if one of the cfg_* arguments has a subset of the default configs, merge them
    cfg_eval_infer = OmegaConf.merge(defaults.eval.infer, cfg_eval_infer)

    n_trajs = cfg_eval_infer.n_trajs
    if n_trajs == -1:
        n_trajs = data_test.num_samples

    if params is not None:
        if state is None:
            state = {}
    else:
        params, state, _, _ = load_haiku(load_ckp)

    key, seed_worker, generator = set_seed(seed)

    loader_test = DataLoader(
        dataset=data_test,
        batch_size=cfg_eval_infer.batch_size,
        collate_fn=numpy_collate,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    metrics_computer = MetricsComputer(
        cfg_eval_infer.metrics,
        dist_fn=case.displacement,
        metadata=data_test.metadata,
        input_seq_length=data_test.input_seq_length,
        stride=cfg_eval_infer.metrics_stride,
    )
    # Precompile model
    model_apply = jit(model.apply)

    # init values
    pos_input_and_target, particle_type = next(iter(loader_test))
    sample = (pos_input_and_target[0], particle_type[0])
    key, _, _, neighbors = case.allocate(key, sample)

    eval_metrics = eval_rollout(
        model_apply=model_apply,
        case=case,
        metrics_computer=metrics_computer,
        params=params,
        state=state,
        neighbors=neighbors,
        loader_eval=loader_test,
        n_rollout_steps=n_rollout_steps,
        n_trajs=n_trajs,
        rollout_dir=rollout_dir,
        out_type=cfg_eval_infer.out_type,
        n_extrap_steps=cfg_eval_infer.n_extrap_steps,
    )
    return eval_metrics
