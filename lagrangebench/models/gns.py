"""Graph Network-based Simulator (Equinox).

Reference: Sanchez-Gonzalez et al. (https://arxiv.org/abs/2002.09405).
"""

from typing import Dict, List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jraph

from lagrangebench.utils import NodeType

from .base import BaseModel
from .utils import Embedding, MLPBlock, build_mlp


class _MPStep(eqx.Module):
    """One Graph Network block: edge update, node update, residual."""

    edge_mlp: MLPBlock
    node_mlp: MLPBlock

    def __init__(self, latent_size: int, blocks_per_step: int, *, key):
        k_e, k_n = jax.random.split(key)
        self.edge_mlp = build_mlp(
            input_size=3 * latent_size,
            latent_size=latent_size,
            output_size=latent_size,
            num_hidden_layers=blocks_per_step,
            is_layer_norm=True,
            key=k_e,
        )
        self.node_mlp = build_mlp(
            input_size=2 * latent_size,
            latent_size=latent_size,
            output_size=latent_size,
            num_hidden_layers=blocks_per_step,
            is_layer_norm=True,
            key=k_n,
        )

    def __call__(
        self,
        nodes: jnp.ndarray,
        edges: jnp.ndarray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        msg_in = jnp.concatenate(
            [nodes[senders], nodes[receivers], edges], axis=-1
        )
        new_edges = self.edge_mlp(msg_in)
        aggr = jraph.segment_sum(new_edges, receivers, nodes.shape[0])
        node_in = jnp.concatenate([nodes, aggr], axis=-1)
        new_nodes = self.node_mlp(node_in)
        return nodes + new_nodes, edges + new_edges


class GNS(BaseModel):
    r"""Graph Network-based Simulator by
    `Sanchez-Gonzalez et al. <https://arxiv.org/abs/2002.09405>`_.
    """

    node_encoder: MLPBlock
    edge_encoder: MLPBlock
    mp_steps: List[_MPStep]
    decoder: MLPBlock
    embedding: Embedding
    output_size: int = eqx.field(static=True)
    num_mp_steps: int = eqx.field(static=True)
    num_particle_types: int = eqx.field(static=True)

    def __init__(
        self,
        particle_dimension: int,
        latent_size: int,
        blocks_per_step: int,
        num_mp_steps: int,
        particle_type_embedding_size: int,
        node_in_size: int,
        edge_in_size: int,
        num_particle_types: int = NodeType.SIZE,
        *,
        key,
    ):
        """Initialize the model.

        Args:
            particle_dimension: Space dimensionality (2 or 3).
            latent_size: Size of the latent representations.
            blocks_per_step: Number of MLP hidden layers per GN block.
            num_mp_steps: Number of message passing steps.
            particle_type_embedding_size: Size of the particle-type embedding.
            node_in_size: Size of scalar node features produced by
                ``case.preprocess`` (sum of vel_hist, vel_mag, bound, force dims).
            edge_in_size: Size of edge features produced by ``case.preprocess``
                (``rel_disp`` + ``rel_dist``).
            num_particle_types: Max number of particle types; embedding is skipped
                when ``num_particle_types <= 1``.
            key: PRNG key for initialization.
        """
        self.output_size = particle_dimension
        self.num_mp_steps = num_mp_steps
        self.num_particle_types = num_particle_types

        k_emb, k_nenc, k_eenc, k_dec, *k_mp = jax.random.split(key, 4 + num_mp_steps)
        self.embedding = Embedding(
            num_particle_types, particle_type_embedding_size, key=k_emb
        )

        embed_size = particle_type_embedding_size if num_particle_types > 1 else 0
        self.node_encoder = build_mlp(
            input_size=node_in_size + embed_size,
            latent_size=latent_size,
            output_size=latent_size,
            num_hidden_layers=blocks_per_step,
            is_layer_norm=True,
            key=k_nenc,
        )
        self.edge_encoder = build_mlp(
            input_size=edge_in_size,
            latent_size=latent_size,
            output_size=latent_size,
            num_hidden_layers=blocks_per_step,
            is_layer_norm=True,
            key=k_eenc,
        )
        self.mp_steps = [
            _MPStep(latent_size, blocks_per_step, key=k) for k in k_mp
        ]
        self.decoder = build_mlp(
            input_size=latent_size,
            latent_size=latent_size,
            output_size=particle_dimension,
            num_hidden_layers=blocks_per_step,
            is_layer_norm=False,
            key=k_dec,
        )

    def _transform(
        self, features: Dict[str, jnp.ndarray]
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        node_features = [
            features[k]
            for k in ["vel_hist", "vel_mag", "bound", "force"]
            if k in features
        ]
        edge_features = [
            features[k] for k in ["rel_disp", "rel_dist"] if k in features
        ]
        nodes = jnp.concatenate(node_features, axis=-1)
        edges = jnp.concatenate(edge_features, axis=-1)
        return nodes, edges, features["senders"], features["receivers"]

    def __call__(
        self, sample: Tuple[Dict[str, jnp.ndarray], jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        features, particle_type = sample
        nodes, edges, senders, receivers = self._transform(features)

        if self.num_particle_types > 1:
            nodes = jnp.concatenate([nodes, self.embedding(particle_type)], axis=-1)

        nodes = self.node_encoder(nodes)
        edges = self.edge_encoder(edges)
        for step in self.mp_steps:
            nodes, edges = step(nodes, edges, senders, receivers)
        return {"acc": self.decoder(nodes)}
