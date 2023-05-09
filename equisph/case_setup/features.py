import enum
from typing import Callable, Dict, List, Optional

import jax
import jax.numpy as jnp
from jax import lax, vmap
from jax_md import partition, space


# TODO remove this before submission
class NodeType(enum.IntEnum):
    FLUID = 0
    SOLID_WALL = 1
    MOVING_WALL = 2
    RIGID_BODY = 3
    SIZE = 9


# TODO remove this before submission
def get_kinematic_mask(particle_type):
    """Returns a boolean mask, set to true for all kinematic (obstacle)
    particles"""
    return jnp.logical_or(
        particle_type == NodeType.SOLID_WALL, particle_type == NodeType.MOVING_WALL
    )


def physical_feature_builder(
    bounds: list,
    normalization_stats: dict,
    connectivity_radius: float,
    displacement_fn: Callable,
    pbc: List[bool],
    magnitudes: bool = False,
    external_force_fn: Optional[Callable] = None,
) -> Callable:
    """Builds a physical feature transform function.

    Transform raw coordinates to
        - Historical velocity sequence
        - Velocity magnitudes
        - Distance to boundaries
        - Relative displacement vectors

    Args:
        bounds: Each sublist contains the lower and upper bound of a dimension.
        normalization_stats: Dict containing mean and std of velocities and targets
        connectivity_radius: Radius of the connectivity graph.
        displacement_fn: Displacement function.
        pbc: Wether to use periodic boundary conditions.
        magnitudes: Whether to include the magnitude of the velocity.
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

        features["vel_hist"] = flat_velocity_sequence

        if magnitudes:
            # append the magnitude of the velocity of each particle to the node features
            velocity_magnitude_sequence = jnp.linalg.norm(
                normalized_velocity_sequence, axis=-1
            )
            features["vel_mag"] = velocity_magnitude_sequence

        # TODO remove this before submission (and docstrings)
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

        # Relative displacement and distances normalized to radius (E, 2)
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


def add_gns_noise(key, pos_input, particle_type, input_seq_length, noise_std, shift_fn):
    isl = input_seq_length
    # add noise to the input and adjust the target accordingly
    key, pos_input_noise = _get_random_walk_noise_for_pos_sequence(
        key, pos_input, noise_std_last_step=noise_std
    )
    kinematic_mask = get_kinematic_mask(particle_type)
    pos_input_noise = jnp.where(kinematic_mask[:, None, None], 0.0, pos_input_noise)
    # adjust targets based on the noise from the last input position
    n_potential_targets = pos_input_noise[:, isl:].shape[1]
    pos_target_noise = pos_input_noise[:, isl - 1][:, None, :]
    pos_target_noise = jnp.tile(pos_target_noise, (1, n_potential_targets, 1))
    pos_input_noise = pos_input_noise.at[:, isl:].set(pos_target_noise)

    shift_vmap = vmap(shift_fn, in_axes=(0, 0))
    shift_dvmap = vmap(shift_vmap, in_axes=(0, 0))
    pos_input_noisy = shift_dvmap(pos_input, pos_input_noise)

    return key, pos_input_noisy


def _get_random_walk_noise_for_pos_sequence(
    key, position_sequence, noise_std_last_step
):
    """Returns random-walk noise in the velocity applied to the position.
    Same functionality as above implemented in JAX."""
    key, subkey = jax.random.split(key)
    velocity_sequence_shape = list(position_sequence.shape)
    velocity_sequence_shape[1] -= 1
    n_velocities = velocity_sequence_shape[1]

    velocity_sequence_noise = jax.random.normal(
        subkey, shape=tuple(velocity_sequence_shape)
    )
    velocity_sequence_noise *= noise_std_last_step / (n_velocities**0.5)
    velocity_sequence_noise = jnp.cumsum(velocity_sequence_noise, axis=1)

    position_sequence_noise = jnp.concatenate(
        [
            jnp.zeros_like(velocity_sequence_noise[:, 0:1]),
            jnp.cumsum(velocity_sequence_noise, axis=1),
        ],
        axis=1,
    )

    return key, position_sequence_noise
