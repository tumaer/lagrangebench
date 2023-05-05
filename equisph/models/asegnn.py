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
    if len(attributes.shape) > 2:
        n_vels = attributes.shape[1]
        attribute_emb = e3nn.concatenate(
            [attributes[:, i, :] for i in range(n_vels)], 1
        ).regroup()
    else:
        attribute_emb = attributes

    # steer either with ones or with the last attribute
    right = attributes[:, -1, :] if right_attribute else None
    left_irreps = attribute_emb.irreps
    right_irreps = right.irreps if right else None
    # NOTE: no biases in the embedding
    for i in range(blocks - 1):
        attribute_emb = O3TensorProductGate(
            hidden_irreps,
            left_irreps=left_irreps,
            right_irreps=right_irreps,
            biases=False,
            name=f"{where}_attribute_attention_{name}_{i}",
            init_fn=avg_initialization,
        )(attribute_emb, right)
    return O3TensorProduct(
        embed_irreps,
        left_irreps=left_irreps,
        right_irreps=right_irreps,
        biases=False,
        name=f"{where}_attribute_attention_{name}",
        init_fn=avg_initialization,
    )(attribute_emb, right)


def AttributeAttention(
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

    def _embed(st_graph: SteerableGraphsTuple) -> Tuple[jnp.ndarray, jnp.ndarray]:
        node_attributes_emb = HistoryEmbeddingBlock(
            st_graph.node_attributes,
            name,
            "node",
            embed_irreps,
            right_attribute,
            hidden_irreps,
            blocks,
        )
        if embed_msg_features:
            edge_attributes_emb = HistoryEmbeddingBlock(
                st_graph.node_attributes,
                name,
                "edge",
                embed_irreps,
                right_attribute,
                hidden_irreps,
                blocks,
            )
        else:
            edge_attributes_emb = st_graph.edge_attributes
        return node_attributes_emb, edge_attributes_emb

    return _embed


def AttentionEmbedding(
    attribute_irreps: Irreps,
    hidden_irreps: Irreps,
    attribute_attention_blocks: int = 2,
    right_attribute: bool = False,
    embed_msg_features: bool = False,
):
    def _embed(st_graph: SteerableGraphsTuple) -> SteerableGraphsTuple:
        # attribute embedding
        node_attributes_emb, edge_attributes_emb = AttributeAttention(
            attribute_irreps,
            "attribute_embedding",
            blocks=attribute_attention_blocks,
            hidden_irreps=hidden_irreps,
            right_attribute=right_attribute,
            embed_msg_features=embed_msg_features,
        )(st_graph)
        # node embedding
        nodes = O3TensorProduct(
            hidden_irreps,
            left_irreps=st_graph.graph.nodes.irreps,
            right_irreps=attribute_irreps,
            name="node_embedding",
        )(st_graph.graph.nodes, node_attributes_emb)
        # edge embedding
        if embed_msg_features:
            additional_message_features = O3TensorProduct(
                hidden_irreps,
                left_irreps=st_graph.additional_message_features.irreps,
                right_irreps=attribute_irreps,
                name="msg_features_embedding",
            )(st_graph.additional_message_features, edge_attributes_emb)
        else:
            additional_message_features = st_graph.additional_message_features

        return st_graph._replace(
            graph=st_graph.graph._replace(nodes=nodes),
            node_attributes=node_attributes_emb,
            edge_attributes=edge_attributes_emb,
            additional_message_features=additional_message_features,
        )

    return _embed


def AttentionDecoder(
    attribute_irreps: Irreps,
    output_irreps: Irreps,
    hidden_irreps: Irreps,
    blocks: int = 1,
    attribute_embedding_blocks: int = 1,
    right_attribute: bool = False,
    embed_msg_features: bool = False,
):
    def _decode(st_graph: SteerableGraphsTuple) -> jnp.ndarray:
        nodes = st_graph.graph.nodes
        node_attributes_emb, _ = AttributeAttention(
            attribute_irreps,
            "decoder",
            blocks=attribute_embedding_blocks,
            hidden_irreps=hidden_irreps,
            right_attribute=right_attribute,
            embed_msg_features=embed_msg_features,
        )(st_graph)
        for i in range(blocks):
            nodes = O3TensorProductGate(
                hidden_irreps,
                left_irreps=nodes.irreps,
                right_irreps=attribute_irreps,
                name=f"decode_{i}",
            )(nodes, node_attributes_emb)

        return O3TensorProduct(
            output_irreps,
            left_irreps=nodes.irreps,
            right_irreps=attribute_irreps,
            name="output",
        )(nodes, node_attributes_emb)

    return _decode


class AttentionSEGNN(SEGNN):
    """Steerable E(3) equivariant GNN with attention mechanism on historical attributes.

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
        ), "SEGNN with attention is supposed to have all historical velocities."

        # network
        self._latent_attribute_irreps = Irreps.spherical_harmonics(
            lmax_latent_attribute
        )
        self._right_attribute = right_attribute
        self._attribute_embedding_blocks = attribute_embedding_blocks
        self._embedding = AttentionEmbedding(
            self._latent_attribute_irreps,
            hidden_irreps=self._hidden_irreps,
            attribute_attention_blocks=self._attribute_embedding_blocks,
            right_attribute=self._right_attribute,
            embed_msg_features=self._embed_msg_features,
        )
        self._decoder = AttentionDecoder(
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
        st_graph = self._embedding(st_graph)

        # message passing
        for n in range(self._num_layers):
            # layer attribute attributes
            node_attributes_emb, edge_attributes_emb = AttributeAttention(
                self._latent_attribute_irreps,
                f"layer_attention_{n}",
                blocks=self._attribute_embedding_blocks,
                hidden_irreps=self._output_irreps,
                right_attribute=self._right_attribute,
                embed_msg_features=self._embed_msg_features,
            )(st_graph)
            st_graph = st_graph._replace(
                node_attributes=node_attributes_emb, edge_attributes=edge_attributes_emb
            )
            st_graph = SEGNNLayer(self._hidden_irreps, n, norm=self._norm)(st_graph)

        return jnp.squeeze(self._decoder(st_graph).array)
