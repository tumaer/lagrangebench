from typing import Dict, Optional, Tuple

import e3nn_jax as e3nn
import haiku as hk
import jax.numpy as jnp
from e3nn_jax import Irreps, IrrepsArray

from .segnn import SEGNN, O3TensorProduct, O3TensorProductGate, SEGNNLayer
from .utils import SteerableGraphsTuple


def avg_initialization(
    name: str,
    path_shape: Tuple[int, ...],
    weight_std: float,
    dtype: jnp.dtype = jnp.float32,
):
    # initialize all params with averaged
    return hk.get_parameter(
        name,
        shape=path_shape,
        dtype=dtype,
        init=hk.initializers.RandomNormal(
            weight_std, 1 / jnp.prod(jnp.array(path_shape))
        ),
    )


def HistoryEmbeddingBlock(
    attributes: IrrepsArray,
    name: str,
    where: str,
    embed_irreps: IrrepsArray,
    right_attribute: bool = False,
    hidden_irreps: Optional[Irreps] = None,
    blocks: int = 1,
):
    """Embed the historical node and edge attributes."""
    if len(attributes.shape) > 2:
        n_vels = attributes.shape[1]
        attribute_emb = e3nn.concatenate(
            [attributes[:, i, :] for i in range(n_vels)], 1
        ).regroup()
    else:
        attribute_emb = attributes

    # steer either with ones or with the last attribute
    right = attributes[:, -1, :] if right_attribute else None
    # NOTE: no biases in the embedding
    for i in range(blocks - 1):
        attribute_emb = O3TensorProductGate(
            hidden_irreps,
            biases=False,
            name=f"{where}_attribute_embedding_{name}_{i}",
            init_fn=avg_initialization,
        )(attribute_emb, right)
    return O3TensorProduct(
        embed_irreps,
        biases=False,
        name=f"{where}_attribute_embedding_{name}",
        init_fn=avg_initialization,
    )(attribute_emb, right)


def AttributeEmbedding(
    embed_irreps: Irreps,
    name: str,
    hidden_irreps: Optional[Irreps],
    blocks: int = 2,
    right_attribute: bool = False,
    embed_msg_features: bool = False,
):
    """Embed the historical attributes of the nodes and edges.
    Args:
        embed_irreps: Attrubte embedding irreps
        name: Name of the embedding
        hidden_irreps: Irreps of the embedding hidden layer
        blocks: Number of hidden layers
        right_attribute: Whether to use the last attribute as the right input
        embed_msg_features: Whether to embed the message features
    """

    if blocks > 0:
        assert hidden_irreps is not None, "Hidden irreps must be specified"

    def _embed(
        node_attributes: IrrepsArray, edge_attributes: IrrepsArray
    ) -> Tuple[IrrepsArray, IrrepsArray]:
        node_attributes_emb = HistoryEmbeddingBlock(
            node_attributes,
            name,
            "node",
            embed_irreps,
            right_attribute,
            hidden_irreps,
            blocks,
        )
        if embed_msg_features:
            edge_attributes_emb = HistoryEmbeddingBlock(
                node_attributes,
                name,
                "edge",
                embed_irreps,
                right_attribute,
                hidden_irreps,
                blocks,
            )
        else:
            edge_attributes_emb = edge_attributes
        return node_attributes_emb, edge_attributes_emb

    return _embed


def HAEGraphEmbedding(
    attribute_irreps: Irreps,
    hidden_irreps: Irreps,
    attribute_embedding_blocks: int = 2,
    right_attribute: bool = False,
    embed_msg_features: bool = False,
):
    """Node (and edge) embedding based on historical attributes."""

    def _embed(
        st_graph: SteerableGraphsTuple,
    ) -> SteerableGraphsTuple:
        # embed attributes for the node embedding
        node_attributes_emb, edge_attributes_emb = AttributeEmbedding(
            attribute_irreps,
            "attribute_embedding",
            blocks=attribute_embedding_blocks,
            hidden_irreps=hidden_irreps,
            right_attribute=right_attribute,
            embed_msg_features=embed_msg_features,
        )(st_graph.node_attributes, st_graph.edge_attributes)
        # node embedding
        nodes = O3TensorProduct(hidden_irreps, name="node_embedding")(
            st_graph.graph.nodes, node_attributes_emb
        )
        # edge embedding
        if embed_msg_features:
            additional_message_features = O3TensorProduct(
                hidden_irreps, name="msg_features_embedding"
            )(st_graph.additional_message_features, edge_attributes_emb)
        else:
            additional_message_features = st_graph.additional_message_features

        return st_graph._replace(
            graph=st_graph.graph._replace(nodes=nodes),
            additional_message_features=additional_message_features,
        )

    return _embed


def HAEGraphDecoder(
    attribute_irreps: Irreps,
    output_irreps: Irreps,
    hidden_irreps: Irreps,
    blocks: int = 1,
    attribute_embedding_blocks: int = 1,
    right_attribute: bool = False,
    embed_msg_features: bool = False,
):
    """Decoder based on historical attributes."""

    def _decode(st_graph: SteerableGraphsTuple) -> jnp.ndarray:
        nodes = st_graph.graph.nodes
        node_attributes_emb, _ = AttributeEmbedding(
            attribute_irreps,
            "decoder",
            blocks=attribute_embedding_blocks,
            hidden_irreps=hidden_irreps,
            right_attribute=right_attribute,
            embed_msg_features=embed_msg_features,
        )(st_graph.node_attributes, st_graph.edge_attributes)
        for i in range(blocks):
            nodes = O3TensorProductGate(hidden_irreps, name=f"decode_{i}")(
                nodes, node_attributes_emb
            )

        return O3TensorProduct(output_irreps, name="output")(nodes, node_attributes_emb)

    return _decode


class HAESEGNN(SEGNN):
    """Steerable E(3) equivariant GNN with historical attribute embedding (HAE).

    Args:
        lmax_latent_attribute: Maximum L of the attribute latent space.
        right_attribute: Whether to use the last attribute as the right input.
        attribute_embedding_blocks: Number of hidden layers in the attribute embedding.
    """

    def __init__(
        self,
        *args,
        lmax_latent_attribute: int = 1,
        right_attribute: bool = False,
        attribute_embedding_blocks: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # transform
        assert (
            self._velocity_aggregate == "all"
        ), "SEGNN with attribute embedding is supposed to have all past velocities."

        # network
        self._latent_attribute_irreps = Irreps.spherical_harmonics(
            lmax_latent_attribute
        )
        self._right_attribute = right_attribute
        self._attribute_embedding_blocks = attribute_embedding_blocks
        self._embedding = HAEGraphEmbedding(
            self._latent_attribute_irreps,
            hidden_irreps=self._hidden_irreps,
            attribute_embedding_blocks=self._attribute_embedding_blocks,
            right_attribute=self._right_attribute,
            embed_msg_features=self._embed_msg_features,
        )
        self._decoder = HAEGraphDecoder(
            self._latent_attribute_irreps,
            self._output_irreps,
            hidden_irreps=self._hidden_irreps,
            attribute_embedding_blocks=self._attribute_embedding_blocks,
            right_attribute=self._right_attribute,
            embed_msg_features=self._embed_msg_features,
        )

    def __call__(
        self, sample: Tuple[Dict[str, jnp.ndarray], jnp.ndarray]
    ) -> jnp.ndarray:
        st_graph = self._transform(*sample)
        # keep full attributes for attribute embedding
        node_attributes_full = IrrepsArray(
            st_graph.node_attributes.irreps, st_graph.node_attributes.array.copy()
        )
        edge_attributes_full = IrrepsArray(
            st_graph.edge_attributes.irreps, st_graph.edge_attributes.array.copy()
        )
        st_graph = self._embedding(st_graph)

        # message passing
        for n in range(self._num_layers):
            # layer attribute attributes
            node_attributes_emb, edge_attributes_emb = AttributeEmbedding(
                self._latent_attribute_irreps,
                f"layer_embedding_{n}",
                blocks=self._attribute_embedding_blocks,
                hidden_irreps=self._output_irreps,
                right_attribute=self._right_attribute,
                embed_msg_features=self._embed_msg_features,
            )(node_attributes_full, edge_attributes_full)
            st_graph = st_graph._replace(
                node_attributes=node_attributes_emb,
                edge_attributes=edge_attributes_emb,
            )
            # segnn layer itself is unchanged
            st_graph = SEGNNLayer(self._hidden_irreps, n, norm=self._norm)(st_graph)

        # reset full attributes for decoder
        st_graph = st_graph._replace(
            node_attributes=node_attributes_full,
            edge_attributes=edge_attributes_full,
        )
        return jnp.squeeze(self._decoder(st_graph).array)
