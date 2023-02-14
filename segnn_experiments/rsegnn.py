from typing import Callable

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


def HistoryEmbedding(
    embed_irreps: e3nn.Irreps,
    aggregate_fn: Callable = e3nn.mean,
    embed_msg_features: bool = False,
):
    def _embed(st_graph: SteerableGraphsTuple) -> SteerableGraphsTuple:
        embeded_st_graph = st_graph._replace(
            graph=st_graph.graph._replace(
                nodes=O3TensorProduct(
                    x=st_graph.graph.nodes,
                    y=aggregate_fn(st_graph.node_attributes, 1),
                    output_irreps=embed_irreps,
                    name="o3_embedding_nodes",
                )
            )
        )
        if embed_msg_features:
            return embeded_st_graph._replace(
                graph=st_graph.graph._replace(
                    edges=O3TensorProduct(
                        st_graph.graph.edges,
                        aggregate_fn(st_graph.edge_attributes, 1)
                        if len(st_graph.edge_attributes.shape) > 2
                        else st_graph.edge_attributes,
                        embed_irreps,
                        name="o3_embedding_edges",
                    )
                )
            )
        return embeded_st_graph

    return _embed


def HistoryDecoder(
    latent_irreps: e3nn.Irreps,
    output_irreps: e3nn.Irreps,
    aggregate: str = "mean",
    blocks: int = 1,
):

    assert aggregate in ["mean", "sum", "last"]

    def _decode(st_graph: SteerableGraphsTuple) -> jnp.ndarray:

        if aggregate == "mean":
            attributes = e3nn.mean(st_graph.node_attributes, 1)
        elif aggregate == "sum":
            attributes = e3nn.sum(st_graph.node_attributes, 1)
        elif aggregate == "last":
            attributes = st_graph.node_attributes[:, -1, :]

        nodes = st_graph.graph.nodes

        for i in range(blocks):
            nodes = O3TensorProductGate(
                nodes,
                attributes,
                latent_irreps,
                name=f"decode_{i}",
            )

        return jnp.squeeze(
            O3TensorProduct(
                nodes,
                attributes,
                output_irreps,
                name="output",
            ).array
        )

    return _decode


class RSEGNN(SEGNN):
    """Steerable E(3) equivariant network with historical attribute rewind.

    Each historical attribute is used in a different message passing step.
    """

    def __init__(self, *args, **kwargs):
        super(RSEGNN, self).__init__(*args, **kwargs)  # noqa # pylint: disable=R1725

        self._output_irreps = kwargs.get("output_irreps")

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
        # TODO: this is unreadable
        return st_graph._replace(
            graph=jraph.GraphNetwork(
                update_node_fn=Partial(
                    update_fn, st_graph.node_attributes[:, layer_num, :]
                ),
                update_edge_fn=Partial(
                    message_fn,
                    st_graph.edge_attributes[:, layer_num, :]
                    if len(st_graph.edge_attributes.shape) > 2
                    else st_graph.edge_attributes,
                    st_graph.additional_message_features,
                ),
                aggregate_edges_for_nodes_fn=jraph.segment_sum,
            )(st_graph.graph)
        )

    def __call__(self, st_graph: SteerableGraphsTuple) -> jnp.array:
        assert st_graph.node_attributes.shape[1] == len(self._hidden_irreps_units)
        if len(st_graph.edge_attributes.shape) > 2:
            assert st_graph.edge_attributes.shape[1] == len(self._hidden_irreps_units)
        # embedding
        # NOTE: embed with average velocity
        st_graph = HistoryEmbedding(
            self._hidden_irreps_units[0],
            aggregate_fn=e3nn.mean,
            embed_msg_features=self._embed_msg_features,
        )(st_graph)

        # message passing layers
        for n, hrp in enumerate(self._hidden_irreps_units):
            st_graph = self._propagate(st_graph, irreps=hrp, layer_num=n)

        # NOTE: decode with last attribute
        return HistoryDecoder(
            self._hidden_irreps_units[-1], self._output_irreps, aggregate="last"
        )(st_graph)
