import os
import pickle
import time
import warnings
from typing import Iterable, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
from torch.utils.data import DataLoader

from equisph.evaluate.metrics import MetricsComputer, MetricsDict, averaged_metrics
from equisph.case_setup import CaseSetupFn, get_kinematic_mask
from equisph.utils import broadcast_from_batch, write_vtk


def eval_single_rollout(
    case: CaseSetupFn,
    metrics_computer: MetricsComputer,
    model_apply: hk.TransformedWithState,
    params: hk.Params,
    state: hk.State,
    neighbors: jnp.ndarray,
    traj_i: Tuple[jnp.ndarray, jnp.ndarray],
    n_rollout_steps: int,
    t_window: int,
    n_extrap_steps: int = 0,
) -> Tuple[jnp.ndarray, MetricsDict, jnp.ndarray]:
    pos_input, particle_type = traj_i
    # (n_nodes, t_window, dim)
    initial_positions = pos_input[:, 0:t_window]
    traj_len = n_rollout_steps + n_extrap_steps
    # (n_nodes, traj_len - t_window, dim)
    ground_truth_positions = pos_input[:, t_window : t_window + traj_len]
    current_positions = initial_positions  # (n_nodes, t_window, dim)
    n_nodes, _, dim = ground_truth_positions.shape
    predictions = jnp.zeros((n_rollout_steps + n_extrap_steps, n_nodes, dim))

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
        # TODO update state in evaluation?
        normalized_acc, _ = model_apply(params, state, (features, particle_type))

        next_position = case.integrate(normalized_acc, current_positions)

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
    case: CaseSetupFn,
    metrics_computer: MetricsComputer,
    model_apply: hk.TransformedWithState,
    params: hk.Params,
    state: hk.State,
    neighbors: jnp.ndarray,
    loader_eval: Iterable,
    n_rollout_steps: int,
    n_trajs: int,
    rollout_dir: str,
    out_type: str = "none",
    n_extrap_steps: int = 0,
) -> Tuple[MetricsDict, jnp.ndarray]:
    t_window = loader_eval.dataset.input_seq_length
    eval_metrics = {}

    if rollout_dir is not None:
        os.makedirs(rollout_dir, exist_ok=True)

    for i, traj_i in enumerate(loader_eval):
        # remove batch dimension
        assert traj_i[0].shape[0] == 1, "Batch dimension should be 1"
        traj_i = broadcast_from_batch(traj_i, index=0)  # (nodes, t, dim)

        example_rollout, metrics, neighbors = eval_single_rollout(
            case=case,
            metrics_computer=metrics_computer,
            model_apply=model_apply,
            params=params,
            state=state,
            neighbors=neighbors,
            traj_i=traj_i,
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

    return eval_metrics, neighbors


def infer(
    model: hk.TransformedWithState,
    params: hk.Params,
    state: hk.State,
    neighbors: jnp.ndarray,
    loader_eval: DataLoader,
    case: CaseSetupFn,
    metrics_computer: MetricsComputer,
    args,
):
    model_apply = jax.jit(model.apply)
    eval_metrics, _ = eval_rollout(
        case=case,
        metrics_computer=metrics_computer,
        model_apply=model_apply,
        params=params,
        state=state,
        neighbors=neighbors,
        loader_eval=loader_eval,
        n_rollout_steps=args.config.n_rollout_steps,
        n_trajs=args.config.eval_n_trajs,
        rollout_dir=args.config.rollout_dir,
        out_type=args.config.out_type,
        n_extrap_steps=args.config.n_extrap_steps,
    )
    print(averaged_metrics(eval_metrics))
