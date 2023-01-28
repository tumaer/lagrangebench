from typing import Callable, Tuple

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import jax.tree_util as tree
import jraph
import numpy as np
from segnn_jax import SteerableGraphsTuple

from gns_jax.utils import NodeType


def steerable_graph_transform_builder(
    node_features_irreps: e3nn.Irreps,
    edge_features_irreps: e3nn.Irreps,
    lmax_attributes: int,
    pbc: Tuple[bool, bool, bool],
    velocity_aggregate: str = "avg",
    attribute_mode: str = "add",
) -> Callable:
    """
    Convert the standard gns GraphsTuple into a SteerableGraphsTuple to use in SEGNN.
    """

    attribute_irreps = e3nn.Irreps.spherical_harmonics(lmax_attributes)

    assert velocity_aggregate in ["avg", "sum", "last", "all"]
    assert attribute_mode in ["velocity", "add", "concat"]

    # TODO: for now only 1) all directions periodic or 2) none of them
    if np.array(pbc).any():  # if PBC, no boundary forces
        num_boundary_entries = 0
    else:
        num_boundary_entries = 6

    def graph_transform(
        graph: jraph.GraphsTuple,
        particle_type: jnp.ndarray,
    ) -> SteerableGraphsTuple:
        # remove the last two bounary displacement vectors
        n_vels = (graph.nodes.shape[1] - num_boundary_entries) // 3
        traj = jnp.reshape(
            graph.nodes[..., : 3 * n_vels],
            (graph.nodes.shape[0], n_vels, 3),
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

        rel_pos = graph.edges[..., :3]

        edge_attributes = e3nn.spherical_harmonics(
            attribute_irreps, rel_pos, normalize=True, normalization="integral"
        )
        vel_embedding = e3nn.spherical_harmonics(
            attribute_irreps, vel, normalize=True, normalization="integral"
        )
        # scatter edge attributes
        sum_n_node = tree.tree_leaves(graph.nodes)[0].shape[0]

        if attribute_mode == "velocity":
            node_attributes = vel_embedding
        else:
            scattered_edges = tree.tree_map(
                lambda e: jraph.segment_mean(e, graph.receivers, sum_n_node),
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

        return SteerableGraphsTuple(
            graph=jraph.GraphsTuple(
                nodes=e3nn.IrrepsArray(
                    node_features_irreps,
                    jnp.concatenate(
                        [graph.nodes, jax.nn.one_hot(particle_type, NodeType.SIZE)],
                        axis=-1,
                    ),
                ),
                edges=None,
                senders=graph.senders,
                receivers=graph.receivers,
                n_node=graph.n_node,
                n_edge=graph.n_edge,
                globals=graph.globals,
            ),
            node_attributes=node_attributes,
            edge_attributes=edge_attributes,
            additional_message_features=e3nn.IrrepsArray(
                edge_features_irreps, graph.edges
            ),
        )

    return graph_transform
