from typing import Tuple

import e3nn_jax as e3nn
import jax.numpy as jnp
import jraph
from jax.tree_util import Partial
from segnn_jax import (
    SEGNN,
    O3TensorProduct,
    O3TensorProductGate,
    SEGNNLayer,
    SteerableGraphsTuple,
)


def AttentionEmbedding(
    embed_irreps: e3nn.Irreps,
    name: str,
    embed_msg_features: bool = False,
):
    def _embed(st_graph: SteerableGraphsTuple) -> Tuple[jnp.ndarray, jnp.ndarray]:
        n_vels = st_graph.node_attributes.shape[1]
        node_attributes_flat = e3nn.concatenate(
            [st_graph.node_attributes[:, i, :] for i in range(n_vels)], 1
        ).regroup()
        # NOTE: no biases here
        right = e3nn.IrrepsArray(
            f"{node_attributes_flat.irreps.dim}x0e",
            jnp.ones_like(node_attributes_flat.array),
        )
        node_attributes_emb = O3TensorProduct(
            x=node_attributes_flat,
            y=right,
            output_irreps=embed_irreps,
            biases=False,
            name=f"node_attribute_embedding_{name}",
        )
        if embed_msg_features:
            if len(st_graph.edge_attributes.shape) > 2:
                n_vels = st_graph.node_attributes.shape[1]
                edge_attributes_flat = e3nn.concatenate(
                    [st_graph.edge_attributes[:, i, :] for i in range(n_vels)], 1
                ).regroup()
            else:
                edge_attributes_flat = st_graph.edge_attributes
            edges_attributes_emb = O3TensorProduct(
                x=edge_attributes_flat,
                y=right,
                output_irreps=embed_irreps,
                biases=False,
                name=f"edge_attribute_embedding_{name}",
            )
        else:
            edges_attributes_emb = st_graph.edge_attributes
        return node_attributes_emb, edges_attributes_emb

    return _embed


def AttentionDecoder(
    latent_irreps: e3nn.Irreps,
    embedding_latent_irreps: e3nn.Irreps,
    output_irreps: e3nn.Irreps,
    embed_msg_features: bool = False,
    blocks: int = 1,
):
    def _decode(st_graph: SteerableGraphsTuple) -> jnp.ndarray:
        nodes = st_graph.graph.nodes
        node_attributes_emb, _ = AttentionEmbedding(
            embedding_latent_irreps, "decoder", embed_msg_features
        )(st_graph)
        for i in range(blocks):
            nodes = O3TensorProductGate(
                nodes,
                node_attributes_emb,
                latent_irreps,
                name=f"decode_{i}",
            )

        return jnp.squeeze(
            O3TensorProduct(
                nodes,
                node_attributes_emb,
                output_irreps,
                name="output",
            ).array
        )

    return _decode


class AttentionSEGNN(SEGNN):
    """Steerable E(3) equivariant network with attention mechanism.

    Args:
        lmax_latent: Maximum spherical harmonics level of the attribute latent space
    """

    def __init__(self, *args, lmax_latent: int = 1, **kwargs):
        super(AttentionSEGNN, self).__init__(  # noqa # pylint: disable=R1725
            *args, **kwargs
        )
        self._output_irreps = kwargs.get("output_irreps")
        self._embedding_latent_irreps = e3nn.Irreps.spherical_harmonics(lmax_latent)

    def _propagate(
        self, st_graph: SteerableGraphsTuple, irreps: e3nn.Irreps, layer_num: int
    ) -> SteerableGraphsTuple:
        """Perform a message passing step by using the historical attributes.
        Args:
            st_graph: Input graph
            irreps: Irreps in the hidden layer
            layer_num: Numbering of the layer
        Returns:
            The updated graph
        """

        message_fn, update_fn = SEGNNLayer(
            output_irreps=irreps,
            layer_num=layer_num,
            blocks=self._blocks_per_layer,
            norm=self._norm,
        )

        # layer attribute attributes
        node_attributes_emb, edge_attributes_emb = AttentionEmbedding(
            self._embedding_latent_irreps, "encoder", self._embed_msg_features
        )(st_graph)

        return st_graph._replace(
            graph=jraph.GraphNetwork(
                update_node_fn=Partial(update_fn, node_attributes_emb),
                update_edge_fn=Partial(
                    message_fn,
                    edge_attributes_emb,
                    st_graph.additional_message_features,
                ),
                aggregate_edges_for_nodes_fn=jraph.segment_sum,
            )(st_graph.graph)
        )

    def __call__(self, st_graph: SteerableGraphsTuple) -> jnp.array:
        # initial attribute embedding
        node_attributes_emb, edge_attributes_emb = AttentionEmbedding(
            self._embedding_latent_irreps, "encoder", self._embed_msg_features
        )(st_graph)

        nodes = O3TensorProduct(
            st_graph.graph.nodes,
            node_attributes_emb,
            self._hidden_irreps_units[0],
            name="node_embedding",
        )
        st_graph = st_graph._replace(graph=st_graph.graph._replace(nodes=nodes))

        if self._embed_msg_features:
            additional_message_features = O3TensorProduct(
                st_graph.additional_message_features,
                edge_attributes_emb,
                self._hidden_irreps_units[0],
                name="msg_features_embedding",
            )
            st_graph = st_graph._replace(
                additional_message_features=additional_message_features
            )

        # message passing layers
        for n, hrp in enumerate(self._hidden_irreps_units):
            st_graph = self._propagate(st_graph, irreps=hrp, layer_num=n)

        return AttentionDecoder(
            latent_irreps=self._hidden_irreps_units[-1],
            embedding_latent_irreps=self._embedding_latent_irreps,
            output_irreps=self._output_irreps,
            embed_msg_features=self._embed_msg_features,
        )(st_graph)
