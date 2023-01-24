import argparse
import enum
import os
import pickle
from typing import Callable  # Dict, List, Optional, Tuple, Union

import cloudpickle
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import pyvista
from jax import lax, vmap
from jax_md import dataclasses, partition, space

# Physical setup for the GNS model.


class NodeType(enum.IntEnum):
    FLUID = 0
    SOLID_WALL = 1
    MOVING_WALL = 2
    RIGID_BODY = 3
    SIZE = 9


def graph_transform_builder(
    bounds: list,
    normalization_stats: dict,
    connectivity_radius: float,
    displacement_fn: Callable,
) -> Callable:
    """Convert raw coordinates to jraph GraphsTuple."""

    def graph_transform(
        pos_input: jnp.ndarray,
        nbrs: partition.NeighborList,
    ) -> jraph.GraphsTuple:
        """Convert raw coordinates to jraph GraphsTuple."""

        n_total_points = pos_input.shape[0]
        most_recent_position = pos_input[:, -1]  # (n_nodes, 2)
        velocity_sequence = pos_input[:, 1:] - pos_input[:, :-1]
        # senders and receivers are integers of shape (E,)
        senders, receivers = nbrs.idx
        node_features = []
        # Normalized velocity sequence, merging spatial an time axis.
        velocity_stats = normalization_stats["velocity"]
        normalized_velocity_sequence = (
            velocity_sequence - velocity_stats["mean"]
        ) / velocity_stats["std"]
        flat_velocity_sequence = normalized_velocity_sequence.reshape(
            n_total_points, -1
        )
        node_features.append(flat_velocity_sequence)

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
        node_features.append(normalized_clipped_distance_to_boundaries)

        # Collect edge features.
        edge_features = []

        # Relative displacement and distances normalized to radius
        # (E, 2)
        displacement = vmap(displacement_fn)(
            most_recent_position[senders], most_recent_position[receivers]
        )
        normalized_relative_displacements = displacement / connectivity_radius
        edge_features.append(normalized_relative_displacements)

        normalized_relative_distances = space.distance(
            normalized_relative_displacements
        )
        edge_features.append(normalized_relative_distances[:, None])

        return jraph.GraphsTuple(
            nodes=jnp.concatenate(node_features, axis=-1),
            edges=jnp.concatenate(edge_features, axis=-1),
            receivers=receivers,
            senders=senders,
            globals=None,
            n_node=jnp.array([n_total_points]),
            n_edge=jnp.array([len(senders)]),
        )

    return graph_transform


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


def _add_gns_noise(key, pos_input, particle_type, pos_target, noise_std):
    # add noise to the input and adjust the target accordingly
    key, pos_input_noise = _get_random_walk_noise_for_pos_sequence(
        key, pos_input, noise_std_last_step=noise_std
    )
    kinematic_mask = get_kinematic_mask(particle_type)
    pos_input_noise = jnp.where(kinematic_mask[:, None, None], 0.0, pos_input_noise)
    pos_input_noisy = pos_input + pos_input_noise
    pos_target_adjusted = pos_target + pos_input_noise[:, -1]
    return key, pos_input_noisy, pos_target_adjusted


@dataclasses.dataclass
class SetupFn:
    allocate: Callable = dataclasses.static_field()
    preprocess: Callable = dataclasses.static_field()
    allocate_eval: Callable = dataclasses.static_field()
    preprocess_eval: Callable = dataclasses.static_field()
    integrate: Callable = dataclasses.static_field()


def setup_builder(args: argparse.Namespace):
    """Contains essentially everything except the model itself.

    Very much inspired by the `partition.neighbor_list` function in JAX-MD.

    The core functions are:
        allocate - allocate memory for the neighbors list
        preprocess - update the neighbors list
        integrate - Semi-implicit Euler respecting periodic boundary conditions
    """

    normalization_stats = args.normalization_stats

    # apply PBC in all directions or not at all
    if np.array(args.metadata["periodic_boundary_conditions"]).any():
        displacement_fn, shift_fn = space.periodic(side=args.box)
    else:
        displacement_fn, shift_fn = space.free()

    neighbor_fn = partition.neighbor_list(
        displacement_fn,
        args.box,
        r_cutoff=args.metadata["default_connectivity_radius"],
        dr_threshold=args.metadata["default_connectivity_radius"] * 0.25,
        capacity_multiplier=1.25,
        mask_self=False,
        format=partition.Sparse,
    )

    graph_transform = graph_transform_builder(
        bounds=args.metadata["bounds"],
        normalization_stats=normalization_stats,
        connectivity_radius=args.metadata["default_connectivity_radius"],
        displacement_fn=displacement_fn,
    )

    def _compute_target(pos_input, pos_target):
        current_velocity = pos_input[:, -1] - pos_input[:, -2]
        next_velocity = pos_target - pos_input[:, -1]
        current_acceleration = next_velocity - current_velocity
        acc_stats = normalization_stats["acceleration"]
        normalized_acceleration = (
            current_acceleration - acc_stats["mean"]
        ) / acc_stats["std"]
        return normalized_acceleration

    def _preprocess(
        sample,
        neighbors=None,
        is_allocate=False,
        mode="train",
        **kwargs,  # key, noise_std
    ):

        if mode == "train":
            pos_input, particle_type, pos_target = sample
            pos_input, pos_target = jnp.array(pos_input), jnp.array(pos_target)
            key, noise_std = kwargs["key"], kwargs["noise_std"]
            if pos_input.shape[1] > 1:
                key, pos_input, pos_target = _add_gns_noise(
                    key, pos_input, particle_type, pos_target, noise_std
                )
        elif mode == "eval":
            pos_input, particle_type = sample

        # allocate the neighbor list
        most_recent_position = pos_input[:, -1]
        if is_allocate:
            neighbors = neighbor_fn.allocate(most_recent_position)
        else:
            neighbors = neighbors.update(most_recent_position)

        # encode desired features and generate jraph graph.
        graph = graph_transform(pos_input, neighbors)

        if mode == "train":
            # compute target acceleration. Inverse of postprocessing step.
            normalized_acceleration = _compute_target(pos_input, pos_target)
            return key, graph, normalized_acceleration, neighbors
        elif mode == "eval":
            return graph, neighbors

    def allocate_fn(key, sample, noise_std=0.0):
        return _preprocess(sample, key=key, noise_std=noise_std, is_allocate=True)

    @jax.jit
    def preprocess_fn(key, sample, noise_std, neighbors):
        return _preprocess(sample, neighbors, key=key, noise_std=noise_std)

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
        most_recent_velocity = most_recent_position - position_sequence[:, -2]

        new_velocity = most_recent_velocity + acceleration  # * dt = 1
        # use the shift function by jax-md to compute the new position
        # this way periodic boundary conditions are automatically taken care of
        new_position = shift_fn(most_recent_position, new_velocity)
        return new_position

    return SetupFn(
        allocate_fn, preprocess_fn, allocate_eval_fn, preprocess_eval_fn, integrate_fn
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


def save_pytree(ckpt_dir: str, pytree_obj, name) -> None:
    with open(os.path.join(ckpt_dir, f"{name}_array.npy"), "wb") as f:
        for x in jax.tree_leaves(pytree_obj):
            np.save(f, x, allow_pickle=False)

    tree_struct = jax.tree_map(lambda t: 0, pytree_obj)
    with open(os.path.join(ckpt_dir, f"{name}_tree.pkl"), "wb") as f:
        pickle.dump(tree_struct, f)


def save_haiku(ckpt_dir: str, params, state, opt_state, step) -> None:
    """https://github.com/deepmind/dm-haiku/issues/18"""

    print("Saving model to", ckpt_dir, "at step", step)

    save_pytree(ckpt_dir, params, "params")
    save_pytree(ckpt_dir, state, "state")

    with open(os.path.join(ckpt_dir, "opt_state.pkl"), "wb") as f:
        cloudpickle.dump(opt_state, f)
    with open(os.path.join(ckpt_dir, "step.txt"), "w") as f:
        f.write(str(step))


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

    with open(os.path.join(model_dir, "step.txt"), "r") as f:
        step = int(f.read())

    return params, state, opt_state, step


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
    setup,
    model_apply,
    params,
    state,
    neighbors,
    traj_i,
    num_rollout_steps,
    input_sequence_length,
    graph_postprocess=None,
):
    pos_input, particle_type = traj_i

    # (n_nodes, t_window, dim)
    initial_positions = pos_input[:, 0:input_sequence_length]
    # (n_nodes, traj_len - t_window, dim)
    ground_truth_positions = pos_input[:, input_sequence_length:]

    current_positions = initial_positions  # (n_nodes, t_window, dim)
    predictions = jnp.zeros_like(ground_truth_positions).transpose(1, 0, 2)

    step = 0
    while step < num_rollout_steps:
        sample = (current_positions, particle_type)
        graph, neighbors = setup.preprocess_eval(sample, neighbors)

        if neighbors.did_buffer_overflow is True:
            edges_ = neighbors.idx.shape
            print(f"(eval) Reallocate neighbors list {edges_} at step {step}")
            _, neighbors = setup.allocate_eval(sample)
            print(f"(eval) To list {neighbors.idx.shape}")

            continue

        if graph_postprocess:
            graph_tuple = graph_postprocess(graph, particle_type)
        else:
            graph_tuple = (graph, particle_type)

        normalized_acceleration, state = model_apply(params, state, graph_tuple)

        next_position = setup.integrate(normalized_acceleration, current_positions)

        kinematic_mask = get_kinematic_mask(particle_type)
        next_position_ground_truth = ground_truth_positions[:, step]

        next_position = jnp.where(
            kinematic_mask[:, None],
            next_position_ground_truth,
            next_position,
        )

        predictions = predictions.at[step].set(next_position)
        current_positions = jnp.concatenate(
            [current_positions[:, 1:], next_position[:, None, :]], axis=1
        )

        step += 1

    # (n_nodes, traj_len - t_window, dim) -> (traj_len - t_window, n_nodes, dim)
    ground_truth_positions = ground_truth_positions.transpose(1, 0, 2)
    loss = ((predictions - ground_truth_positions) ** 2).mean()

    return predictions, loss, neighbors


def eval_rollout(
    setup,
    model_apply,
    params,
    state,
    neighbors,
    loader_valid,
    num_rollout_steps,
    num_trajs,
    rollout_dir,
    is_write_vtk=False,
    graph_postprocess=None,
):

    input_sequence_length = loader_valid.dataset.input_sequence_length
    valid_loss = []
    for i, traj_i in enumerate(loader_valid):
        # remove batch dimension
        assert traj_i[0].shape[0] == 1, "Batch dimension should be 1"
        traj_i = broadcast_from_batch(traj_i, index=0)  # (nodes, t, dim)

        example_rollout, loss, neighbors = eval_single_rollout(
            setup=setup,
            model_apply=model_apply,
            params=params,
            state=state,
            neighbors=neighbors,
            traj_i=traj_i,
            num_rollout_steps=num_rollout_steps,
            input_sequence_length=input_sequence_length,
            graph_postprocess=graph_postprocess,
        )
        valid_loss.append(loss)

        if rollout_dir is not None:
            pos_input = traj_i[0].transpose(1, 0, 2)  # (t, nodes, dim)
            initial_positions = pos_input[:input_sequence_length]
            example_full = np.concatenate([initial_positions, example_rollout], axis=0)
            example_rollout = {
                "predicted_rollout": example_full,  # (t, nodes, dim)
                "ground_truth_rollout": pos_input,  # (t, nodes, dim)
            }
            os.makedirs(rollout_dir, exist_ok=True)

            file_prefix = f"{rollout_dir}/rollout_{i}"
            if is_write_vtk:

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
            else:
                filename = f"{file_prefix}.pkl"

                with open(filename, "wb") as f:
                    pickle.dump(example_rollout, f)

        if (i + 1) == num_trajs:
            break
    return np.mean(valid_loss), neighbors
