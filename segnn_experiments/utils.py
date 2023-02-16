from typing import Callable

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import jax.tree_util as tree
import jraph
from segnn_jax import SteerableGraphsTuple

from gns_jax.utils import NodeType


def segnn_graph_transform_builder(
    node_features_irreps: e3nn.Irreps,
    edge_features_irreps: e3nn.Irreps,
    lmax_attributes: int,
    n_vels: int,
    velocity_aggregate: str = "avg",
    attribute_mode: str = "add",
    homogeneous_particles: bool = False,
) -> Callable:
    """Convert physical features into a SteerableGraphsTuple for SEGNN.

    Parameters:
        node_features_irreps: Irreps of the node features.
        edge_features_irreps: Irreps of the edge features.
        lmax_attributes: Maximum harmonics level of the attributes.
        n_vels: Number of velocities in the history.
        velocity_aggregate: Velocity sequence aggregation method.
        attribute_mode: Node attribute aggregation method.
        homogeneous_particles: If all particles are of homogeneous type.
    """

    attribute_irreps = e3nn.Irreps.spherical_harmonics(lmax_attributes)

    assert velocity_aggregate in ["avg", "sum", "last", "all"]
    assert attribute_mode in ["velocity", "add", "concat"]

    def graph_transform(
        features: jraph.GraphsTuple,
        particle_type: jnp.ndarray,
    ) -> SteerableGraphsTuple:
        """Convert physical features to SteerableGraphsTuple for segnn."""

        assert (
            features["vel_hist"].shape[1] // n_vels == 3
        ), "The velocity history should be of shape (n_nodes, n_vels * 3)."

        traj = jnp.reshape(
            features["vel_hist"], (features["vel_hist"].shape[0], n_vels, 3)
        )

        if n_vels == 1 or velocity_aggregate == "all":
            vel = jnp.squeeze(traj)
        else:
            if velocity_aggregate == "avg":
                vel = jnp.mean(traj, 1)
            if velocity_aggregate == "sum":
                vel = jnp.sum(traj, 1)
            if velocity_aggregate == "last":
                vel = traj[:, -1, :]

        rel_pos = features["rel_disp"]

        edge_attributes = e3nn.spherical_harmonics(
            attribute_irreps, rel_pos, normalize=True, normalization="integral"
        )
        vel_embedding = e3nn.spherical_harmonics(
            attribute_irreps, vel, normalize=True, normalization="integral"
        )
        # scatter edge attributes
        sum_n_node = features["vel_hist"].shape[0]

        if attribute_mode == "velocity":
            node_attributes = vel_embedding
        else:
            scattered_edges = tree.tree_map(
                lambda e: jraph.segment_mean(e, features["receivers"], sum_n_node),
                edge_attributes,
            )
            if attribute_mode == "concat":
                node_attributes = e3nn.concatenate(
                    [scattered_edges, vel_embedding], axis=-1
                )
            if attribute_mode == "add":
                node_attributes = vel_embedding
                # TODO: a bit ugly
                if velocity_aggregate == "all":
                    # transpose for broadcasting
                    node_attributes.array = jnp.transpose(
                        (
                            jnp.transpose(node_attributes.array, (0, 2, 1))
                            + jnp.expand_dims(scattered_edges.array, -1)
                        ),
                        (0, 2, 1),
                    )
                else:
                    node_attributes += scattered_edges

        # scalar attribute to 1 by default
        node_attributes.array = node_attributes.array.at[..., 0].set(1.0)

        node_features = [
            features[k]
            for k in ["vel_hist", "vel_mag", "bound", "force"]
            if k in features
        ]
        node_features = jnp.concatenate(node_features, axis=-1)

        if not homogeneous_particles:
            particles = jax.nn.one_hot(particle_type, NodeType.SIZE)
            node_features = jnp.concatenate([node_features, particles], axis=-1)

        edge_features = [features[k] for k in ["rel_disp", "rel_dist"] if k in features]
        edge_features = jnp.concatenate(edge_features, axis=-1)

        return SteerableGraphsTuple(
            graph=jraph.GraphsTuple(
                nodes=e3nn.IrrepsArray(node_features_irreps, node_features),
                edges=None,
                senders=features["senders"],
                receivers=features["receivers"],
                n_node=jnp.array([sum_n_node]),
                n_edge=jnp.array([len(features["senders"])]),
                globals=None,
            ),
            node_attributes=node_attributes,
            edge_attributes=edge_attributes,
            additional_message_features=e3nn.IrrepsArray(
                edge_features_irreps, edge_features
            ),
        )

    return graph_transform


def node_irreps(args) -> str:
    irreps = []
    irreps.append(f"{args.config.input_seq_length - 1}x1o")
    if not any(args.metadata["periodic_boundary_conditions"]):
        irreps.append("2x1o")

    if args.info.has_external_force:
        irreps.append("1x1o")

    if args.config.magnitudes:
        irreps.append(f"{args.config.input_seq_length - 1}x0e")

    if not args.info.homogeneous_particles:
        irreps.append(f"{NodeType.SIZE}x0e")

    return "+".join(irreps)
