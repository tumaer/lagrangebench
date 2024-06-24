"""Feature extraction utilities."""

from typing import Callable, Dict, List, Optional

import jax
import jax.numpy as jnp
from jax import lax, vmap
from jax_sph.jax_md import partition, space

FeatureDict = Dict[str, jnp.ndarray]
TargetDict = Dict[str, jnp.ndarray]


def physical_feature_builder(
    bounds: list,
    normalization_stats: dict,
    connectivity_radius: float,
    displacement_fn: Callable,
    pbc: List[bool],
    magnitude_features: bool = False,
    external_force_fn: Optional[Callable] = None,
) -> Callable:
    """Build a physical feature transform function.

    Transform raw coordinates to
        - Absolute positions
        - Historical velocity sequence
        - Velocity magnitudes
        - Distance to boundaries
        - External force field
        - Relative displacement vectors and distances

    Args:
        bounds: Each sublist contains the lower and upper bound of a dimension.
        normalization_stats: Dict containing mean and std of velocities and targets
        connectivity_radius: Radius of the connectivity graph.
        displacement_fn: Displacement function.
        pbc: Wether to use periodic boundary conditions.
        magnitude_features: Whether to include the magnitude of the velocity.
        external_force_fn: Function that returns the external force field (optional).
    """
    displacement_fn_vmap = vmap(displacement_fn, in_axes=(0, 0))
    displacement_fn_dvmap = vmap(displacement_fn_vmap, in_axes=(0, 0))

    velocity_stats = normalization_stats["velocity"]

    def feature_transform(
        pos_input: jnp.ndarray,
        nbrs: partition.NeighborList,
    ) -> FeatureDict:
        """Feature engineering.

        Returns:
            Dict of features, with possible keys
                - "abs_pos", absolute positions
                - "vel_hist", historical velocity sequence
                - "vel_mag", velocity magnitudes
                - "bound", distance to boundaries
                - "force", external force field
                - "rel_disp", relative displacement vectors
                - "rel_dist", relative distance vectors
        """
        features = {}

        n_total_points = pos_input.shape[0]
        most_recent_position = pos_input[:, -1]  # (n_nodes, dim)
        # pos_input.shape = (n_nodes, n_timesteps, dim)
        velocity_sequence = displacement_fn_dvmap(pos_input[:, 1:], pos_input[:, :-1])
        # Normalized velocity sequence, merging spatial an time axis.
        normalized_velocity_sequence = (
            velocity_sequence - velocity_stats["mean"]
        ) / velocity_stats["std"]
        flat_velocity_sequence = normalized_velocity_sequence.reshape(
            n_total_points, -1
        )

        features["abs_pos"] = pos_input
        features["vel_hist"] = flat_velocity_sequence

        if magnitude_features:
            # append the magnitude of the velocity of each particle to the node features
            velocity_magnitude_sequence = jnp.linalg.norm(
                normalized_velocity_sequence, axis=-1
            )
            features["vel_mag"] = velocity_magnitude_sequence

        if not any(pbc):
            # Normalized clipped distances to lower and upper boundaries.
            # boundaries are an array of shape [num_dimensions, dim], where the
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
        receivers, senders = nbrs.idx
        features["senders"] = senders
        features["receivers"] = receivers

        # Relative displacement and distances normalized to radius (E, dim)
        displacement = vmap(displacement_fn)(
            most_recent_position[receivers], most_recent_position[senders]
        )
        normalized_relative_displacements = displacement / connectivity_radius
        features["rel_disp"] = normalized_relative_displacements

        normalized_relative_distances = space.distance(
            normalized_relative_displacements
        )
        features["rel_dist"] = normalized_relative_distances[:, None]

        return jax.tree_map(lambda f: f, features)

    return feature_transform
