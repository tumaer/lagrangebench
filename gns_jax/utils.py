import argparse
import enum
import json
import os
import pickle
import time
import warnings
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import cloudpickle
import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import pyvista
from jax import lax, vmap
from jax_md import dataclasses, partition, space

from gns_jax.metrics import MetricsComputer, MetricsDict

# Physical setup for the GNS model.


class NodeType(enum.IntEnum):
    FLUID = 0
    SOLID_WALL = 1
    MOVING_WALL = 2
    RIGID_BODY = 3
    SIZE = 9


def gns_graph_transform_builder() -> Callable:
    """Convert physical features to jraph GraphsTuple for gns."""

    def graph_transform(
        features: Dict[str, jnp.ndarray],
        particle_type: jnp.ndarray,
    ) -> Tuple[jraph.GraphsTuple, jnp.ndarray]:
        """Convert physical features to jraph GraphsTuple for gns."""

        n_total_points = features["vel_hist"].shape[0]
        node_features = [
            features[k]
            for k in ["vel_hist", "vel_mag", "bound", "force"]
            if k in features
        ]
        edge_features = [features[k] for k in ["rel_disp", "rel_dist"] if k in features]

        graph = jraph.GraphsTuple(
            nodes=jnp.concatenate(node_features, axis=-1),
            edges=jnp.concatenate(edge_features, axis=-1),
            receivers=features["receivers"],
            senders=features["senders"],
            n_node=jnp.array([n_total_points]),
            n_edge=jnp.array([len(features["senders"])]),
            globals=None,
        )

        return graph, particle_type

    return graph_transform


def physical_feature_builder(
    bounds: list,
    normalization_stats: dict,
    connectivity_radius: float,
    displacement_fn: Callable,
    pbc: List[bool],
    magnitudes: bool = False,
    log_norm: str = "none",
    external_force_fn: Optional[Callable] = None,
) -> Callable:
    """Feature engineering builder. Transform raw coordinates to
        - Historical velocity sequence
        - Velocity magnitudes
        - Distance to boundaries
        - Relative displacement vectors
    Parameters:
        bounds: Each sublist contains the lower and upper bound of a dimension.
        normalization_stats: Dict containing mean and std of velocities and targets
        connectivity_radius: Radius of the connectivity graph.
        displacement_fn: Displacement function.
        pbc: Wether to use periodic boundary conditions.
        magnitudes: Whether to include the magnitude of the velocity.
        log_norm: Whether to apply log normalization.
        external_force_fn: Function that returns the external force field (optional).
    """

    displacement_fn_vmap = vmap(displacement_fn, in_axes=(0, 0))
    displacement_fn_dvmap = vmap(displacement_fn_vmap, in_axes=(0, 0))

    velocity_stats = normalization_stats["velocity"]

    def feature_transform(
        pos_input: jnp.ndarray,
        nbrs: partition.NeighborList,
    ) -> Dict[str, jnp.ndarray]:
        """Feature engineering.
        Returns:
            Dict of features, with possible keys
                - "vel_hist"
                - "vel_mag"
                - "bound"
                - "force"
                - "rel_disp"
                - "rel_dist"
        """

        features = {}

        n_total_points = pos_input.shape[0]
        most_recent_position = pos_input[:, -1]  # (n_nodes, 2)
        # pos_input.shape = (n_nodes, n_timesteps, dim)
        velocity_sequence = displacement_fn_dvmap(pos_input[:, 1:], pos_input[:, :-1])
        # Normalized velocity sequence, merging spatial an time axis.
        normalized_velocity_sequence = (
            velocity_sequence - velocity_stats["mean"]
        ) / velocity_stats["std"]
        flat_velocity_sequence = normalized_velocity_sequence.reshape(
            n_total_points, -1
        )
        # log normalization
        if log_norm in ["input", "both"]:
            flat_velocity_sequence = log_norm_fn(flat_velocity_sequence)

        features["vel_hist"] = flat_velocity_sequence

        if magnitudes:
            # append the magnitude of the velocity of each particle to the node features
            velocity_magnitude_sequence = jnp.linalg.norm(
                normalized_velocity_sequence, axis=-1
            )
            features["vel_mag"] = velocity_magnitude_sequence
            # node features shape = (n_nodes, (input_seq_length - 1) * (dim + 1))

            # # append the average velocity over all particles to the node features
            # # we hope that this feature can be used like layer normalization
            # vel_mag_seq_mean = velocity_magnitude_sequence.mean(axis=0, keepdims=True)
            # vel_mag_seq_mean_tile = jnp.tile(vel_mag_seq_mean, (n_total_points, 1))
            # node_features.append(vel_mag_seq_mean_tile)

        # TODO: for now just disable it completely if any periodicity applies
        if not any(pbc):
            # Normalized clipped distances to lower and upper boundaries.
            # boundaries are an array of shape [num_dimensions, 2], where the
            # second axis, provides the lower/upper boundaries.
            boundaries = lax.stop_gradient(jnp.array(bounds))

            distance_to_lower_boundary = most_recent_position - boundaries[:, 0][None]
            distance_to_upper_boundary = boundaries[:, 1][None] - most_recent_position

            # rewritten the code above in jax
            distance_to_boundaries = jnp.concatenate(
                [distance_to_lower_boundary, distance_to_upper_boundary], axis=1
            )
            normalized_clipped_distance_to_boundaries = jnp.clip(
                distance_to_boundaries / connectivity_radius, -1.0, 1.0
            )
            features["bound"] = normalized_clipped_distance_to_boundaries

        if external_force_fn is not None:
            external_force_field = vmap(external_force_fn)(most_recent_position)
            features["force"] = external_force_field

        # senders and receivers are integers of shape (E,)
        senders, receivers = nbrs.idx
        features["senders"] = senders
        features["receivers"] = receivers

        # Relative displacement and distances normalized to radius
        # (E, 2)
        displacement = vmap(displacement_fn)(
            most_recent_position[senders], most_recent_position[receivers]
        )
        normalized_relative_displacements = displacement / connectivity_radius
        features["rel_disp"] = normalized_relative_displacements

        normalized_relative_distances = space.distance(
            normalized_relative_displacements
        )
        features["rel_dist"] = normalized_relative_distances[:, None]

        return jax.tree_map(lambda f: f, features)

    return feature_transform


def get_kinematic_mask(particle_type):
    """Returns a boolean mask, set to true for all kinematic (obstacle)
    particles"""
    return jnp.logical_or(
        particle_type == NodeType.SOLID_WALL, particle_type == NodeType.MOVING_WALL
    )


def _get_random_walk_noise_for_pos_sequence(
    key, position_sequence, noise_std_last_step
):
    """Returns random-walk noise in the velocity applied to the position.
    Same functionality as above implemented in JAX."""
    key, subkey = jax.random.split(key)
    velocity_sequence_shape = list(position_sequence.shape)
    velocity_sequence_shape[1] -= 1
    num_velocities = velocity_sequence_shape[1]

    velocity_sequence_noise = jax.random.normal(
        subkey, shape=tuple(velocity_sequence_shape)
    )
    velocity_sequence_noise *= noise_std_last_step / (num_velocities**0.5)
    velocity_sequence_noise = jnp.cumsum(velocity_sequence_noise, axis=1)

    position_sequence_noise = jnp.concatenate(
        [
            jnp.zeros_like(velocity_sequence_noise[:, 0:1]),
            jnp.cumsum(velocity_sequence_noise, axis=1),
        ],
        axis=1,
    )

    return key, position_sequence_noise


def _add_gns_noise(
    key, pos_input, particle_type, input_seq_length, noise_std, shift_fn
):
    isl = input_seq_length
    # add noise to the input and adjust the target accordingly
    key, pos_input_noise = _get_random_walk_noise_for_pos_sequence(
        key, pos_input, noise_std_last_step=noise_std
    )
    kinematic_mask = get_kinematic_mask(particle_type)
    pos_input_noise = jnp.where(kinematic_mask[:, None, None], 0.0, pos_input_noise)
    # adjust targets based on the noise from the last input position
    num_potential_targets = pos_input_noise[:, isl:].shape[1]
    pos_target_noise = pos_input_noise[:, isl - 1][:, None, :]
    pos_target_noise = jnp.tile(pos_target_noise, (1, num_potential_targets, 1))
    pos_input_noise = pos_input_noise.at[:, isl:].set(pos_target_noise)

    shift_vmap = vmap(shift_fn, in_axes=(0, 0))
    shift_dvmap = vmap(shift_vmap, in_axes=(0, 0))
    pos_input_noisy = shift_dvmap(pos_input, pos_input_noise)

    return key, pos_input_noisy


@dataclasses.dataclass
class SetupFn:
    allocate: Callable = dataclasses.static_field()
    preprocess: Callable = dataclasses.static_field()
    allocate_eval: Callable = dataclasses.static_field()
    preprocess_eval: Callable = dataclasses.static_field()
    integrate: Callable = dataclasses.static_field()
    metrics_computer: Callable = dataclasses.static_field()
    displacement: Callable = dataclasses.static_field()


def setup_builder(args: argparse.Namespace, external_force_fn: Callable):
    """Contains essentially everything except the model itself.

    Very much inspired by the `partition.neighbor_list` function in JAX-MD.

    The core functions are:
        allocate - allocate memory for the neighbors list
        preprocess - update the neighbors list
        integrate - Semi-implicit Euler respecting periodic boundary conditions
    """

    dtype = jnp.float64 if args.config.f64 else np.float32
    normalization_stats = args.normalization

    # apply PBC in all directions or not at all
    if np.array(args.metadata["periodic_boundary_conditions"]).any():
        displacement_fn, shift_fn = space.periodic(side=np.array(args.box))
    else:
        displacement_fn, shift_fn = space.free()

    displacement_fn_set = vmap(displacement_fn, in_axes=(0, 0))

    neighbor_fn = partition.neighbor_list(
        displacement_fn,
        np.array(args.box),
        r_cutoff=args.metadata["default_connectivity_radius"],
        capacity_multiplier=1.25,
        mask_self=False,
        format=partition.Sparse,
    )

    feature_transform = physical_feature_builder(
        bounds=args.metadata["bounds"],
        normalization_stats=normalization_stats,
        connectivity_radius=args.metadata["default_connectivity_radius"],
        displacement_fn=displacement_fn,
        pbc=args.metadata["periodic_boundary_conditions"],
        magnitudes=args.config.magnitudes,
        log_norm=args.config.log_norm,
        external_force_fn=external_force_fn,
    )

    input_seq_length = args.config.input_seq_length

    def _compute_target(pos_input):
        # displacement(r1, r2) = r1-r2  # without PBC

        current_velocity = displacement_fn_set(pos_input[:, 1], pos_input[:, 0])
        next_velocity = displacement_fn_set(pos_input[:, 2], pos_input[:, 1])
        current_acceleration = next_velocity - current_velocity

        acc_stats = normalization_stats["acceleration"]
        normalized_acceleration = (
            current_acceleration - acc_stats["mean"]
        ) / acc_stats["std"]

        vel_stats = normalization_stats["velocity"]
        normalized_velocity = (next_velocity - vel_stats["mean"]) / vel_stats["std"]

        return {"acc": normalized_acceleration, "vel": normalized_velocity}

    def _preprocess(
        sample,
        neighbors=None,
        is_allocate=False,
        mode="train",
        **kwargs,  # key, noise_std
    ):

        pos_input = jnp.asarray(sample[0], dtype=dtype)
        particle_type = jnp.asarray(sample[1])

        if mode == "train":
            key, noise_std = kwargs["key"], kwargs["noise_std"]
            unroll_steps = kwargs["unroll_steps"]
            if pos_input.shape[1] > 1:
                key, pos_input = _add_gns_noise(
                    key, pos_input, particle_type, input_seq_length, noise_std, shift_fn
                )

        # allocate the neighbor list
        most_recent_position = pos_input[:, input_seq_length - 1]
        if is_allocate:
            neighbors = neighbor_fn.allocate(most_recent_position)
        else:
            neighbors = neighbors.update(most_recent_position)

        # selected features
        features = feature_transform(pos_input[:, :input_seq_length], neighbors)

        if mode == "train":
            # compute target acceleration. Inverse of postprocessing step.
            # the "-2" is needed because we need the most recent position and one before
            slice_begin = (0, input_seq_length - 2 + unroll_steps, 0)
            slice_size = (pos_input.shape[0], 3, pos_input.shape[2])

            target_dict = _compute_target(
                jax.lax.dynamic_slice(pos_input, slice_begin, slice_size)
            )
            return key, features, target_dict, neighbors
        elif mode == "eval":
            return features, neighbors

    def allocate_fn(key, sample, noise_std=0.0, unroll_steps=0):
        return _preprocess(
            sample,
            key=key,
            noise_std=noise_std,
            unroll_steps=unroll_steps,
            is_allocate=True,
        )

    @jax.jit
    def preprocess_fn(key, sample, noise_std, neighbors, unroll_steps=0):
        return _preprocess(
            sample, neighbors, key=key, noise_std=noise_std, unroll_steps=unroll_steps
        )

    def allocate_eval_fn(sample):
        return _preprocess(sample, is_allocate=True, mode="eval")

    @jax.jit
    def preprocess_eval_fn(sample, neighbors):
        return _preprocess(sample, neighbors, mode="eval")

    @jax.jit
    def integrate_fn(normalized_acceleration, position_sequence):
        """corresponds to `decoder_postprocessor` in the original code."""

        # The model produces the output in normalized space so we apply inverse
        # normalization.
        acceleration_stats = normalization_stats["acceleration"]
        acceleration = acceleration_stats["mean"] + (
            normalized_acceleration * acceleration_stats["std"]
        )

        # Use an Euler integrator to go from acceleration to position, assuming
        # a dt=1 corresponding to the size of the finite difference.
        most_recent_position = position_sequence[:, -1]
        most_recent_velocity = displacement_fn_set(
            most_recent_position, position_sequence[:, -2]
        )

        new_velocity = most_recent_velocity + acceleration  # * dt = 1
        # use the shift function by jax-md to compute the new position
        # this way periodic boundary conditions are automatically taken care of
        new_position = shift_fn(most_recent_position, new_velocity)
        return new_position

    def metrics_computer(predictions, ground_truth):
        return MetricsComputer(
            args.config.metrics,
            displacement_fn,
            args.metadata,
            args.config.input_seq_length,
        )(predictions, ground_truth)

    return SetupFn(
        allocate_fn,
        preprocess_fn,
        allocate_eval_fn,
        preprocess_eval_fn,
        integrate_fn,
        metrics_computer,
        displacement_fn,
    )


# Bathing utilities for JAX pytrees


def broadcast_to_batch(sample, batch_size: int):
    """Broadcast a pytree to a batched one with first dimension batch_size"""
    assert batch_size > 0
    return jax.tree_map(lambda x: jnp.repeat(x[None, ...], batch_size, axis=0), sample)


def broadcast_from_batch(batch, index: int):
    """Broadcast a batched pytree to the sample `index` out of the batch"""
    assert index >= 0
    return jax.tree_map(lambda x: x[index], batch)


# Utilities for saving and loading Haiku models


def save_pytree(ckp_dir: str, pytree_obj, name) -> None:
    with open(os.path.join(ckp_dir, f"{name}_array.npy"), "wb") as f:
        for x in jax.tree_leaves(pytree_obj):
            np.save(f, x, allow_pickle=False)

    tree_struct = jax.tree_map(lambda t: 0, pytree_obj)
    with open(os.path.join(ckp_dir, f"{name}_tree.pkl"), "wb") as f:
        pickle.dump(tree_struct, f)


def save_haiku(ckp_dir: str, params, state, opt_state, metadata_ckp) -> None:
    """https://github.com/deepmind/dm-haiku/issues/18"""

    save_pytree(ckp_dir, params, "params")
    save_pytree(ckp_dir, state, "state")

    with open(os.path.join(ckp_dir, "opt_state.pkl"), "wb") as f:
        cloudpickle.dump(opt_state, f)
    with open(os.path.join(ckp_dir, "metadata_ckp.json"), "w") as f:
        json.dump(metadata_ckp, f)

    if "best" not in ckp_dir:
        ckp_dir_best = os.path.join(ckp_dir, "best")
        metadata_best_path = os.path.join(ckp_dir, "best", "metadata_ckp.json")

        if os.path.exists(metadata_best_path):  # all except first step
            with open(metadata_best_path, "r") as fp:
                metadata_ckp_best = json.loads(fp.read())

            # if loss is better than best previous loss, save to best model directory
            if metadata_ckp["loss"] < metadata_ckp_best["loss"]:
                print(
                    f"Saving model to {ckp_dir} at step {metadata_ckp['step']}"
                    f" with loss {metadata_ckp['loss']} (best so far)"
                )

                save_haiku(ckp_dir_best, params, state, opt_state, metadata_ckp)
        else:  # first step
            save_haiku(ckp_dir_best, params, state, opt_state, metadata_ckp)


def load_pytree(model_dir: str, name):
    """load a pytree from a directory"""
    with open(os.path.join(model_dir, f"{name}_tree.pkl"), "rb") as f:
        tree_struct = pickle.load(f)

    leaves, treedef = jax.tree_flatten(tree_struct)

    with open(os.path.join(model_dir, f"{name}_array.npy"), "rb") as f:
        flat_state = [np.load(f) for _ in leaves]

    return jax.tree_unflatten(treedef, flat_state)


def load_haiku(model_dir: str):
    """https://github.com/deepmind/dm-haiku/issues/18"""

    print("Loading model from", model_dir)

    params = load_pytree(model_dir, "params")
    state = load_pytree(model_dir, "state")

    with open(os.path.join(model_dir, "opt_state.pkl"), "rb") as f:
        opt_state = cloudpickle.load(f)

    with open(os.path.join(model_dir, "metadata_ckp.json"), "r") as fp:
        metadata_ckp = json.loads(fp.read())

    return params, state, opt_state, metadata_ckp["step"]


def get_num_params(params):
    """Get the number of parameters in a Haiku model"""
    return sum(np.prod(p.shape) for p in jax.tree_leaves(params))


def print_params_shapes(params, prefix=""):
    if not isinstance(params, dict):
        print(f"{prefix: <40}, shape = {params.shape}")
    else:
        for k, v in params.items():
            print_params_shapes(v, prefix=prefix + k)


# Utilities for saving generated trajectories


def write_vtk_temp(data_dict, path):
    """Store a .vtk file for ParaView"""
    r = np.asarray(data_dict["r"])
    N, dim = r.shape

    # PyVista treats the position information differently than the rest
    if dim == 2:
        r = np.hstack([r, np.zeros((N, 1))])
    data_pv = pyvista.PolyData(r)

    # copy all the other information also to pyvista, using plain numpy arrays
    for k, v in data_dict.items():
        # skip r because we already considered it above
        if k == "r":
            continue

        # working in 3D or scalar features do not require special care
        if dim == 2 and v.ndim == 2:
            v = np.hstack([v, np.zeros((N, 1))])

        data_pv[k] = np.asarray(v)

    data_pv.save(path)


# Evaluate trained models


def eval_single_rollout(
    setup: SetupFn,
    model_apply: hk.TransformedWithState,
    params: hk.Params,
    state: hk.State,
    neighbors: jnp.ndarray,
    traj_i: Tuple[jnp.ndarray, jnp.ndarray],
    num_rollout_steps: int,
    input_seq_length: int,
    graph_preprocess: Callable,
    eval_n_more_steps: int = 0,
    oversmooth_norm_hops: int = 0,
) -> Dict:
    pos_input, particle_type = traj_i

    # (n_nodes, t_window, dim)
    initial_positions = pos_input[:, 0:input_seq_length]
    # (n_nodes, traj_len - t_window, dim)
    ground_truth_positions = pos_input[:, input_seq_length:]

    current_positions = initial_positions  # (n_nodes, t_window, dim)

    if eval_n_more_steps == 0:
        # the number of predictions is the number of ground truth positions
        predictions = jnp.zeros_like(ground_truth_positions).transpose(1, 0, 2)
    else:
        num_predictions = num_rollout_steps + eval_n_more_steps
        num_nodes, _, dim = ground_truth_positions.shape
        predictions = jnp.zeros((num_predictions, num_nodes, dim))

    step = 0
    while step < num_rollout_steps + eval_n_more_steps:
        sample = (current_positions, particle_type)
        features, neighbors = setup.preprocess_eval(sample, neighbors)

        if neighbors.did_buffer_overflow is True:
            edges_ = neighbors.idx.shape
            print(f"(eval) Reallocate neighbors list {edges_} at step {step}")
            _, neighbors = setup.allocate_eval(sample)
            print(f"(eval) To list {neighbors.idx.shape}")

            continue

        graph = graph_preprocess(features, particle_type)

        # TODO oversmooth for SEGNN  (now fails because of graph.graph)
        if oversmooth_norm_hops > 0:
            graph, most_recent_vel_magnitude = oversmooth_norm(
                graph, oversmooth_norm_hops, input_seq_length
            )

        # predict
        normalized_acceleration, state = model_apply(params, state, graph)

        if oversmooth_norm_hops > 0:
            normalized_acceleration *= most_recent_vel_magnitude[:, None]

        next_position = setup.integrate(normalized_acceleration, current_positions)

        if eval_n_more_steps == 0:
            kinematic_mask = get_kinematic_mask(particle_type)
            next_position_ground_truth = ground_truth_positions[:, step]

            next_position = jnp.where(
                kinematic_mask[:, None],
                next_position_ground_truth,
                next_position,
            )
        else:
            warnings.warn("kinematic mask is not applied in eval_n_more_steps mode.")

        predictions = predictions.at[step].set(next_position)
        current_positions = jnp.concatenate(
            [current_positions[:, 1:], next_position[:, None, :]], axis=1
        )

        step += 1

    # (n_nodes, traj_len - t_window, dim) -> (traj_len - t_window, n_nodes, dim)
    ground_truth_positions = ground_truth_positions.transpose(1, 0, 2)

    return (
        predictions,
        setup.metrics_computer(predictions, ground_truth_positions),
        neighbors,
    )


def eval_rollout(
    setup: SetupFn,
    model_apply: hk.TransformedWithState,
    params: hk.Params,
    state: hk.State,
    neighbors: jnp.ndarray,
    loader_valid: Iterable,
    num_rollout_steps: int,
    num_trajs: int,
    rollout_dir: str,
    graph_preprocess: Callable,
    out_type: str = "none",
    eval_n_more_steps: int = 0,
    oversmooth_norm_hops: int = 0,
) -> Tuple[MetricsDict, jnp.ndarray]:

    input_seq_length = loader_valid.dataset.input_seq_length
    eval_metrics = {}

    if rollout_dir is not None:
        os.makedirs(rollout_dir, exist_ok=True)

    for i, traj_i in enumerate(loader_valid):
        # remove batch dimension
        assert traj_i[0].shape[0] == 1, "Batch dimension should be 1"
        traj_i = broadcast_from_batch(traj_i, index=0)  # (nodes, t, dim)

        example_rollout, metrics, neighbors = eval_single_rollout(
            setup=setup,
            model_apply=model_apply,
            params=params,
            state=state,
            neighbors=neighbors,
            traj_i=traj_i,
            num_rollout_steps=num_rollout_steps,
            input_seq_length=input_seq_length,
            graph_preprocess=graph_preprocess,
            eval_n_more_steps=eval_n_more_steps,
            oversmooth_norm_hops=oversmooth_norm_hops,
        )

        eval_metrics[f"rollout_{i}"] = metrics

        if rollout_dir is not None:
            pos_input = traj_i[0].transpose(1, 0, 2)  # (t, nodes, dim)
            initial_positions = pos_input[:input_seq_length]
            example_full = np.concatenate([initial_positions, example_rollout], axis=0)
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
                    write_vtk_temp(state_vtk, filename_vtk)

                for j in range(pos_input.shape[0]):
                    filename_vtk = file_prefix + f"_ref_{j}.vtk"
                    state_vtk = {
                        "r": example_rollout["ground_truth_rollout"][j],
                        "tag": traj_i[1],
                    }
                    write_vtk_temp(state_vtk, filename_vtk)
            if out_type == "pkl":
                filename = f"{file_prefix}.pkl"

                with open(filename, "wb") as f:
                    pickle.dump(example_rollout, f)

        if (i + 1) == num_trajs:
            break

    if rollout_dir is not None:
        # save metrics
        t = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        with open(f"{rollout_dir}/metrics{t}.pkl", "wb") as f:
            pickle.dump(eval_metrics, f)

    return eval_metrics, neighbors


def averaged_metrics(eval_metrics: MetricsDict) -> Dict[str, float]:
    """Averages the metrics over the rollouts."""
    small_metrics = defaultdict(lambda: 0.0)
    for rollout in eval_metrics.values():
        for k, m in rollout.items():
            if k == "e_kin":
                continue
            if k in ["mse", "mae"]:
                k = "loss"
            small_metrics[f"val/{k}"] += float(jnp.mean(m)) / len(eval_metrics)
    return dict(small_metrics)


# Unit Test utils


class Linear(hk.Module):
    """Model defining linear relation between input nodes and targets.

    Used as unit test case.
    """

    def __init__(self, dim_out):
        super().__init__()
        self.mlp = hk.nets.MLP([dim_out], activate_final=False, name="MLP")

    def __call__(
        self, input_: Tuple[jraph.GraphsTuple, np.ndarray]
    ) -> jraph.GraphsTuple:
        return jax.vmap(self.mlp)(jnp.concatenate(input_, axis=-1))


# Normalization utils


def safe_log(x: jnp.ndarray) -> jnp.ndarray:
    """Logarithm with clipping to avoid numerical issues."""
    return jnp.log(jnp.clip(x, a_min=1e-10, a_max=None))


def log_norm_fn(x: jnp.ndarray) -> jnp.ndarray:
    """Log-normalization for Gaussian-distributed data.

    Design choices:
    1. Clipping is applied to avoid numerical issues.
    2. The value 1.11 guarantees that stardard Gaussian inputs x are mapped to a
    distribution with mean 0 and standard deviation 1 (however, not Gaussian anymore).
    3. The value 0.637 guarantees that if the inputs x are Gaussian with std S, then
    the outputs of this function have the same std as if the outputs had std 1/S.
    """

    return jnp.sign(x) * (safe_log(jnp.abs(x)) + 0.637) / 1.11


def get_dataset_normalization(
    metadata: Dict[str, List[float]],
    is_isotropic_norm: bool,
    noise_std: float,
) -> Dict[str, Dict[str, np.ndarray]]:

    acc_mean = np.array(metadata["acc_mean"])
    acc_std = np.array(metadata["acc_std"])
    vel_mean = np.array(metadata["vel_mean"])
    vel_std = np.array(metadata["vel_std"])

    if is_isotropic_norm:
        warnings.warn(
            "The isotropic normalization is only a simplification of the general case."
            "It is only valid if the means of the velocity and acceleration are"
            "isotropic -> we use $max(abs(mean)) < 1% min(std)$ as a heuristic."
        )

        acc_mean = np.mean(acc_mean) * np.ones_like(acc_mean)
        acc_std = np.sqrt(np.mean(acc_std**2)) * np.ones_like(acc_std)
        vel_mean = np.mean(vel_mean) * np.ones_like(vel_mean)
        vel_std = np.sqrt(np.mean(vel_std**2)) * np.ones_like(vel_std)

    return {
        "acceleration": {
            "mean": acc_mean,
            "std": np.sqrt(acc_std**2 + noise_std**2),
        },
        "velocity": {
            "mean": vel_mean,
            "std": np.sqrt(vel_std**2 + noise_std**2),
        },
    }


def oversmooth_norm(graph, hops, input_seq_length):
    isl = input_seq_length
    # assumes that the last three channels are the most recent velocity
    most_recent_vel = graph.nodes[:, (isl - 2) * 3 : (isl - 1) * 3]
    most_recent_vel_magnitude = jnp.linalg.norm(most_recent_vel, axis=1)

    # average over velocity magnitudes to get an estimate of average velocity.
    for _ in range(hops):
        most_recent_vel_magnitude = jraph.segment_mean(
            most_recent_vel_magnitude[graph.senders],
            graph.receivers,
            graph.nodes.shape[0],
        )

    rescaled_vel = jnp.where(
        most_recent_vel_magnitude[:, None],
        graph.nodes[:, : (isl - 1) * 3] / most_recent_vel_magnitude[:, None],
        0,
    )
    new_node_features = graph.nodes.at[:, : (isl - 1) * 3].set(rescaled_vel)
    graph = graph._replace(nodes=new_node_features)

    return graph, most_recent_vel_magnitude


# push-forward utils


def push_forward_sample_steps(key, step, pushforward):
    key, key_unroll = jax.random.split(key, 2)

    # steps needs to be an ordered list
    steps = np.array(pushforward["steps"])
    assert all(steps[i] <= steps[i + 1] for i in range(len(steps) - 1))

    # until which index to sample from
    idx = (step > steps).sum()

    unroll_steps = jax.random.choice(
        key_unroll,
        a=jnp.array(pushforward["unrolls"][:idx]),
        p=jnp.array(pushforward["probs"][:idx]),
    )
    return key, unroll_steps


def push_forward_build(graph_preprocess, model_apply, setup):
    @jax.jit
    def push_forward(features, current_pos, particle_type, neighbors, params, state):
        # no buffer overflow check here, since push forward acts on later epochs
        graph = graph_preprocess(features, particle_type)
        normalized_acceleration, _ = model_apply(params, state, graph)
        next_pos = setup.integrate(normalized_acceleration, current_pos)
        current_pos = jnp.concatenate(
            [current_pos[:, 1:], next_pos[:, None, :]], axis=1
        )

        features, neighbors = setup.preprocess_eval(
            (current_pos, particle_type), neighbors
        )
        return current_pos, neighbors, features

    return push_forward


# Reproducibility utils


def set_seed_from_config(seed):
    """Set seeds for numpy, random and torch."""

    import random

    import torch

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # those below are not used by the jax models
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # dataloader-related seeds
    def seed_worker(_):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return seed_worker, generator
