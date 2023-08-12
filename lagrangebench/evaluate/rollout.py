"""Evaluation and inference functions for generating rollouts."""

import os
import pickle
import time
import warnings
from typing import Callable, Dict, Iterable, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jax_md.partition as partition
from torch.utils.data import DataLoader

from lagrangebench.data import H5Dataset
from lagrangebench.data.utils import numpy_collate
from lagrangebench.defaults import defaults
from lagrangebench.evaluate.metrics import MetricsComputer, MetricsDict
from lagrangebench.utils import (
    broadcast_from_batch,
    get_kinematic_mask,
    load_haiku,
    set_seed,
    write_vtk,
)


def eval_single_rollout(
    model_apply: Callable,
    case,
    params: hk.Params,
    state: hk.State,
    traj_i: Tuple[jnp.ndarray, jnp.ndarray],
    neighbors: partition.NeighborList,
    metrics_computer: MetricsComputer,
    n_rollout_steps: int,
    t_window: int,
    n_extrap_steps: int = 0,
) -> Tuple[jnp.ndarray, MetricsDict, jnp.ndarray]:
    """Compute the rollout on a single trajectory.

    Args:
        model_apply: Model function.
        case: CaseSetupFn class.
        params: Haiku params.
        state: Haiku state.
        traj_i: Trajectory to evaluate.
        neighbors: Neighbor list.
        metrics_computer: MetricsComputer with the desired metrics.
        n_rollout_steps: Number of rollout steps.
        t_window: Length of the input sequence.
        n_extrap_steps: Number of extrapolation steps (beyond the ground truth rollout).

    Returns:
        A tuple with (predicted rollout, metrics, neighbor list).
    """
    pos_input, particle_type = traj_i
    # if n_rollout_steps set to -1, use the whole trajectory
    if n_rollout_steps < 0:
        n_rollout_steps = pos_input.shape[1] - t_window

    initial_positions = pos_input[:, 0:t_window]  # (n_nodes, t_window, dim)
    traj_len = n_rollout_steps + n_extrap_steps  # (n_nodes, traj_len - t_window, dim)
    ground_truth_positions = pos_input[:, t_window : t_window + traj_len]
    current_positions = initial_positions  # (n_nodes, t_window, dim)
    n_nodes, _, dim = ground_truth_positions.shape

    predictions = jnp.zeros((traj_len, n_nodes, dim))

    step = 0
    while step < n_rollout_steps + n_extrap_steps:
        sample = (current_positions, particle_type)
        features, neighbors = case.preprocess_eval(sample, neighbors)

        if neighbors.did_buffer_overflow is True:
            edges_ = neighbors.idx.shape
            print(f"(eval) Reallocate neighbors list {edges_} at step {step}")
            _, neighbors = case.allocate_eval(sample)
            print(f"(eval) To list {neighbors.idx.shape}")

            continue

        # predict
        pred, _ = model_apply(params, state, (features, particle_type))

        next_position = case.integrate(pred, current_positions)

        if n_extrap_steps == 0:
            kinematic_mask = get_kinematic_mask(particle_type)
            next_position_ground_truth = ground_truth_positions[:, step]

            next_position = jnp.where(
                kinematic_mask[:, None],
                next_position_ground_truth,
                next_position,
            )
        else:
            warnings.warn("kinematic mask not applied in extrapolation mode.")

        predictions = predictions.at[step].set(next_position)
        current_positions = jnp.concatenate(
            [current_positions[:, 1:], next_position[:, None, :]], axis=1
        )

        step += 1

    # (n_nodes, traj_len - t_window, dim) -> (traj_len - t_window, n_nodes, dim)
    ground_truth_positions = ground_truth_positions.transpose(1, 0, 2)

    return (
        predictions,
        metrics_computer(predictions, ground_truth_positions),
        neighbors,
    )


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
    t_window = loader_eval.dataset.input_seq_length
    eval_metrics = {}

    if rollout_dir is not None:
        os.makedirs(rollout_dir, exist_ok=True)

    for i, traj_i in enumerate(loader_eval):
        # remove batch dimension
        assert traj_i[0].shape[0] == 1, "Batch dimension should be 1"
        traj_i = broadcast_from_batch(traj_i, index=0)  # (nodes, t, dim)

        example_rollout, metrics, neighbors = eval_single_rollout(
            model_apply=model_apply,
            case=case,
            params=params,
            state=state,
            traj_i=traj_i,
            neighbors=neighbors,
            metrics_computer=metrics_computer,
            n_rollout_steps=n_rollout_steps,
            t_window=t_window,
            n_extrap_steps=n_extrap_steps,
        )

        eval_metrics[f"rollout_{i}"] = metrics

        if rollout_dir is not None:
            pos_input = traj_i[0].transpose(1, 0, 2)  # (t, nodes, dim)
            initial_positions = pos_input[:t_window]
            example_full = jnp.concatenate([initial_positions, example_rollout], axis=0)
            example_rollout = {
                "predicted_rollout": example_full,  # (t, nodes, dim)
                "ground_truth_rollout": pos_input,  # (t, nodes, dim)
            }

            file_prefix = f"{rollout_dir}/rollout_{i}"
            if out_type == "vtk":
                for j in range(pos_input.shape[0]):
                    filename_vtk = file_prefix + f"_{j}.vtk"
                    state_vtk = {
                        "r": example_rollout["predicted_rollout"][j],
                        "tag": traj_i[1],
                    }
                    write_vtk(state_vtk, filename_vtk)

                for j in range(pos_input.shape[0]):
                    filename_vtk = file_prefix + f"_ref_{j}.vtk"
                    state_vtk = {
                        "r": example_rollout["ground_truth_rollout"][j],
                        "tag": traj_i[1],
                    }
                    write_vtk(state_vtk, filename_vtk)
            if out_type == "pkl":
                filename = f"{file_prefix}.pkl"

                with open(filename, "wb") as f:
                    pickle.dump(example_rollout, f)

        if (i + 1) == n_trajs:
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
    dataset_test: H5Dataset,
    params: Optional[hk.Params] = None,
    state: Optional[hk.State] = None,
    load_checkpoint: Optional[str] = None,
    metrics: Optional[Dict] = None,
    rollout_dir: Optional[str] = None,
    eval_n_trajs: int = defaults.eval_n_trajs,
    n_rollout_steps: int = defaults.n_rollout_steps,
    out_type: str = defaults.out_type,
    n_extrap_steps: int = defaults.n_extrap_steps,
    seed: int = defaults.seed,
):
    """
    Infer on a dataset, compute metrics and optionally save rollout in out_type format.

    Args:
        model: (Transformed) Haiku model.
        case: Case setup class.
        dataset_test: Test dataset.
        params: Haiku params.
        state: Haiku state.
        load_checkpoint: Path to checkpoint directory.
        metrics: Metrics to compute.
        rollout_dir: Path to rollout directory.
        eval_n_trajs: Number of trajectories to evaluate.
        n_rollout_steps: Number of rollout steps.
        out_type: Output type. Either "none", "vtk" or "pkl".
        n_extrap_steps: Number of extrapolation steps.
        seed: Seed.

    Returns:
        eval_metrics: Metrics per trajectory.
    """
    assert (
        params is not None or load_checkpoint is not None
    ), "Either params or a load_checkpoint directory must be provided for inference."

    if params is not None:
        if state is None:
            state = {}
    else:
        params, state, _, _ = load_haiku(load_checkpoint)

    key, seed_worker, generator = set_seed(seed)

    loader_test = DataLoader(
        dataset=dataset_test,
        batch_size=1,
        collate_fn=numpy_collate,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    metrics_computer = MetricsComputer(
        metrics,
        dist_fn=case.displacement,
        metadata=dataset_test.metadata,
        input_seq_length=dataset_test.input_seq_length,
    )
    # Precompile model
    model_apply = jax.jit(model.apply)

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
        n_trajs=eval_n_trajs,
        rollout_dir=rollout_dir,
        out_type=out_type,
        n_extrap_steps=n_extrap_steps,
    )
    return eval_metrics
