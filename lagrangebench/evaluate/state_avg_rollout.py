"""Evaluation and inference functions for generating rollouts."""
"""
Here, for every initial condition, we compute 5 different random initial 
purely noised state and then denoise them to find the acc/vel and then 
average them before performing the time integration. 
"""

import os
import pickle
import time
from functools import partial
from typing import Callable, Iterable, List, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jax_md.partition as partition
from jax import jit, lax, random, vmap
from torch.utils.data import DataLoader

from lagrangebench.data import H5Dataset
from lagrangebench.data.utils import get_dataset_stats, numpy_collate
from lagrangebench.defaults import defaults
from lagrangebench.evaluate.metrics import MetricsComputer, MetricsDict
from lagrangebench.evaluate.utils import write_vtk
import os
from lagrangebench.utils import (
    ACDMConfig,
    broadcast_from_batch,
    broadcast_to_batch,
    get_kinematic_mask,
    load_haiku,
    set_seed,
)


# For ACDM:
@partial(
    jit,
    static_argnames=[
        "model_apply",
        "case_integrate",
        "displacement_fn_set",
        "acdm_config",
        "input_seq_length",
        "metrics_computer",
    ],
)
def _forward_eval_acdm(
    key_s: int,
    params: hk.Params,
    state: hk.State,
    sample: Tuple[jnp.ndarray, jnp.ndarray],
    current_positions: jnp.ndarray,
    target_positions: jnp.ndarray,
    model_apply: Callable,
    case_integrate: Callable,
    displacement_fn_set: Callable,
    acdm_config: ACDMConfig,
    noise_std: float,
    input_seq_length: int,
    metrics_computer: MetricsComputer,
) -> jnp.ndarray:
    """Run one update of the 'current_state' using the trained model"""

    features, particle_type = sample

    normalization_stats = get_dataset_stats(
        metrics_computer.__dict__["_metadata"],
        is_isotropic_norm=False,
        noise_std=noise_std,
    )
    displacement_fn_set = vmap(displacement_fn_set, in_axes=(0, 0))

    def compute_acc_based_on_pos_slice(pos_input: jnp.ndarray):
        """Compute acceleration based on position slice."""
        current_velocity = displacement_fn_set(pos_input[:, 1], pos_input[:, 0])
        next_velocity = displacement_fn_set(pos_input[:, 2], pos_input[:, 1])
        current_acceleration = next_velocity - current_velocity
        acc_stats = normalization_stats["acceleration"]
        return (current_acceleration - acc_stats["mean"]) / acc_stats["std"]

    def compute_vel_based_on_pos_slice(pos_input: jnp.ndarray):
        """Compute velocity based on position slice."""
        current_velocity = displacement_fn_set(pos_input[:, 1], pos_input[:, 0])
        vel_stats = normalization_stats["velocity"]
        return (current_velocity - vel_stats["mean"]) / vel_stats["std"]

    def extract_conditioning_data(
        pos_input: jnp.ndarray, num_conditioning_steps: int, conditioning_parameter: str
    ):
        """Extract conditioning data for ACDM."""
        conditioning_data = {}
        if conditioning_parameter == "acc":
            assert input_seq_length >= num_conditioning_steps + 2
        elif conditioning_parameter == "vel":
            assert input_seq_length >= num_conditioning_steps + 1

        for i in range(num_conditioning_steps):
            if conditioning_parameter == "acc":
                slice_begin = (
                    0,
                    input_seq_length - num_conditioning_steps - 2 + i,
                    0,
                )  # (0,2,0)
                slice_size = (pos_input.shape[0], 3, pos_input.shape[2])  # (3200,3,2)
                value = compute_acc_based_on_pos_slice(
                    lax.dynamic_slice(pos_input, slice_begin, slice_size)
                )
                key = f"acc_t_minus_{num_conditioning_steps - i}"
                conditioning_data[key] = value

            elif conditioning_parameter == "vel":
                slice_begin = (0, input_seq_length - num_conditioning_steps - 1 + i, 0)
                slice_size = (pos_input.shape[0], 2, pos_input.shape[2])
                value = compute_vel_based_on_pos_slice(
                    lax.dynamic_slice(pos_input, slice_begin, slice_size)
                )
                key = f"vel_t_minus_{num_conditioning_steps - i}"
                conditioning_data[key] = value

        return conditioning_data

    conditioning_data = extract_conditioning_data(
        features["abs_pos"],
        acdm_config.num_conditioning_steps,
        acdm_config.conditioning_parameter,
    )
    # concatenates the values of the dictionary, need to convert to list first
    conditioning_data = jnp.concatenate(list(conditioning_data.values()), axis=1)

    # conditioning data only without the target
    features["prev_concatenated_data"] = conditioning_data
    de_Noised = jnp.zeros((key_s.shape[0],features["vel_hist"].shape[0], 2))
    
    for i in range(key_s.shape[0]): #loop over different keys (seeds) and then find the average

        # dNoise has a shape (3200,2)
        dNoise = random.normal(key_s[i], jnp.zeros((features["vel_hist"].shape[0], 2)).shape)
        
        key, subkey = random.split(key_s[i],2)
        # cNoise has a shape (3200,4)
        cNoise = random.normal(
            subkey,
            jnp.zeros(
                (features["vel_hist"].shape[0], 2 * acdm_config.num_conditioning_steps)
            ).shape,
        )

        for k in reversed(range(0, acdm_config.diffusionSteps)):  # Refinement loop
            # compute conditioned part with normal forward diffusion
            condNoisy = (
                acdm_config.sqrtAlphasCumprod[k] * features["prev_concatenated_data"]
                + acdm_config.sqrtOneMinusAlphasCumprod[k] * cNoise
            )

            features["k"] = jnp.tile(k, (features["vel_hist"].shape[0],))
            features["noised_data"] = jnp.concatenate([condNoisy, dNoise], axis=1)

            pred, state = model_apply(params, state, (features, particle_type))

            modelMean = acdm_config.sqrtRecipAlphas[k] * (
                features["noised_data"]
                - acdm_config.betas[k]
                * pred["noise"]
                / acdm_config.sqrtOneMinusAlphasCumprod[k]
            )

            dNoise = modelMean[:, cNoise.shape[1] : modelMean.shape[1]]

            if k != 0:
                # sample randomly (only for non-final prediction),
                # use mean directly for final prediction
                key, subkey = random.split(key,2)
                dNoise = dNoise + acdm_config.sqrtPosteriorVariance[k] * random.normal(
                    subkey, jnp.zeros((features["vel_hist"].shape[0], 2)).shape
                )
        
        #replace the zeros in d_Noised with the dNoise which is supposed to be close 
        #to the ground truth acceleration 
        de_Noised = de_Noised.at[i].set(dNoise)
    
    #calculate the average across the batch
    dNoise = jnp.mean(de_Noised, axis=0)
        
    if acdm_config.conditioning_parameter == "acc":
        avg_refined_value = {"acc": dNoise}

    elif acdm_config.conditioning_parameter == "vel":
        avg_refined_value = {"vel": dNoise}

    next_position = case_integrate(avg_refined_value, current_positions)

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



def eval_batched_rollout_acdm(
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
    key_s:int,
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
    # This is the autoregressive rollout loop
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
            key_s,
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


def eval_rollout_acdm_state_avg(
    model_apply: Callable,
    case,
    params: hk.Params,
    state: hk.State,
    loader_eval: Iterable,
    neighbors: partition.NeighborList,
    metrics_computer: MetricsComputer,
    n_rollout_steps: int,
    n_trajs: int,
    key: jnp.ndarray,
    acdm_config: ACDMConfig,
    noise_std: float,
    input_seq_length: int,
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

    forward_eval_acdm = partial(
        _forward_eval_acdm,
        model_apply=model_apply,
        case_integrate=case.integrate,
        displacement_fn_set=case.displacement,
        acdm_config=acdm_config,
        noise_std=noise_std,
        input_seq_length=input_seq_length,
        metrics_computer=metrics_computer,
    )
    forward_eval_vmap = vmap(forward_eval_acdm, in_axes=(0, None, None, 0, 0, 0))
    preprocess_eval_vmap = vmap(case.preprocess_eval_acdm, in_axes=(0, 0))
    metrics_computer_vmap = vmap(metrics_computer, in_axes=(0, 0))

    for i, traj_batch_i in enumerate(loader_eval): #len(loader_eval) = 384
        # if n_trajs is not a multiple of batch_size, we slice from the last batch
        n_traj_left = n_trajs - i * batch_size
        if n_traj_left < batch_size:
            traj_batch_i = jax.tree_map(lambda x: x[:n_traj_left], traj_batch_i)

        # numpy to jax
        traj_batch_i = jax.tree_map(lambda x: jnp.array(x), traj_batch_i)
        # (pos_input_batch, particle_type_batch) = traj_batch_i
        # pos_input_batch.shape = (batch, num_particles, seq_length, dim)
        
        (
            example_rollout_batch,
            metrics_batch,
            neighbors,
        ) = eval_batched_rollout_acdm(
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
            key_s=key, #(batch, key)
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
                if out_type == "pkl":
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


def infer_with_state_avg_at_every_step(
    model: hk.TransformedWithState,
    case,
    data_test: H5Dataset,
    params: Optional[hk.Params] = None,
    state: Optional[hk.State] = None,
    load_checkpoint: Optional[str] = None,
    metrics: List = ["mse"],
    rollout_dir: Optional[str] = None,
    eval_n_trajs: int = defaults.eval_n_trajs,
    n_rollout_steps: int = defaults.n_rollout_steps,
    out_type: str = defaults.out_type,
    n_extrap_steps: int = defaults.n_extrap_steps,
    seed: int = defaults.seed,
    metrics_stride: int = defaults.metrics_stride,
    batch_size: int = defaults.batch_size_infer,
    **kwargs,
):
    """
    Infer on a dataset, compute metrics and optionally save rollout in out_type format.

    Args:
        model: (Transformed) Haiku model.
        case: Case setup class.
        data_test: Test dataset.
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
        metrics_stride: Stride for e_kin and sinkhorn.
        batch_size: Batch size for inference.

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

    base_key, seed_worker, generator = set_seed(seed)

    loader_test = DataLoader(
        dataset=data_test,
        batch_size=batch_size,
        collate_fn=numpy_collate,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    metrics_computer = MetricsComputer(
        metrics,
        dist_fn=case.displacement,
        metadata=data_test.metadata,
        input_seq_length=data_test.input_seq_length,
        stride=metrics_stride,
    )
    # Precompile model
    model_apply = jit(model.apply)

    # init values
    pos_input_and_target, particle_type = next(iter(loader_test))
    sample = (pos_input_and_target[0], particle_type[0])

    acdm_config = kwargs["acdm_config"]
    noise_std = kwargs["noise_std"]
    input_seq_length = kwargs["input_seq_length"]

    k = 0 #just used for initialization (allocation)
    key, _, _, neighbors = case.allocate_acdm(base_key, sample, k, acdm_config)
    
    #number of samples = 5 (hardcoded)
    keys = random.split(key, 5)[None, :, :]
    eval_metrics = eval_rollout_acdm_state_avg(
        case=case,
        metrics_computer=metrics_computer,
        model_apply=model_apply,
        params=params,
        state=state,
        neighbors=neighbors,
        loader_eval=loader_test,
        n_rollout_steps=n_rollout_steps,
        n_trajs=eval_n_trajs, #384
        key=keys,
        acdm_config=acdm_config,
        noise_std=noise_std,
        input_seq_length=input_seq_length,
        rollout_dir=rollout_dir,
        out_type=out_type,
    )
    return eval_metrics


