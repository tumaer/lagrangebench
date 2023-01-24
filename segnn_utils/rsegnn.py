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


class RSEGNN(SEGNN):
    """Steerable E(3) equivariant network with historical attributes."""

    def __init__(self, *args, historical_edge_attributes: bool = False, **kwargs):
        super(RSEGNN, self).__init__(*args, **kwargs)  # noqa # pylint: disable=R1725

        self._historical_edge_attributes = historical_edge_attributes
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
                    if self._historical_edge_attributes
                    else st_graph.edge_attributes,
                    st_graph.additional_message_features,
                ),
                aggregate_edges_for_nodes_fn=jraph.segment_sum,
            )(st_graph.graph)
        )

    def __call__(self, st_graph: SteerableGraphsTuple) -> jnp.array:
        assert st_graph.node_attributes.shape[1] == len(self._hidden_irreps_units)
        if self._historical_edge_attributes:
            assert st_graph.edge_attributes.shape[1] == len(self._hidden_irreps_units)
        # embedding
        # NOTE: embed with average velocity
        st_graph = st_graph._replace(
            graph=st_graph.graph._replace(
                nodes=O3TensorProduct(
                    st_graph.graph.nodes,
                    e3nn.mean(st_graph.node_attributes, 1),
                    self._hidden_irreps_units[0],
                    name="o3_embedding_nodes",
                )
            )
        )
        if self._embed_msg_features:
            st_graph = st_graph._replace(
                graph=st_graph.graph._replace(
                    edges=O3TensorProduct(
                        st_graph.graph.edges,
                        e3nn.mean(st_graph.edge_attributes, 1)
                        if self._historical_edge_attributes
                        else st_graph.edge_attributes,
                        self._hidden_irreps_units[0],
                        name="o3_embedding_edges",
                    )
                )
            )

        # message passing layers
        for n, hrp in enumerate(self._hidden_irreps_units):
            st_graph = self._propagate(st_graph, irreps=hrp, layer_num=n)

        # NOTE: decode with last attribute
        # TODO: ugly for now
        nodes = st_graph.graph.nodes
        for i in range(1):
            nodes = O3TensorProductGate(
                nodes,
                st_graph.node_attributes[:, -1, :],
                self._hidden_irreps_units[-1],
                name=f"prepool_{i}",
            )

        return jnp.squeeze(
            O3TensorProduct(
                nodes,
                st_graph.node_attributes[:, -1, :],
                self._output_irreps,
                name="output",
            ).array
        )
