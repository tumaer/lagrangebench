from argparse import Namespace
from typing import Dict, Tuple, Type

import haiku as hk
import jax.numpy as jnp
import jraph

from equisph.case_setup.features import NodeType

from .base import BaseModel
from .utils import build_mlp


class GNS(BaseModel):
    """Model definition for Graph Network-based Simulator (GNS)"""

    def __init__(
        self,
        particle_dimension,
        latent_size,
        num_mlp_layers,
        num_message_passing_steps,
        num_particle_types,
        particle_type_embedding_size,
    ):
        super().__init__()
        self._output_size = particle_dimension
        self._latent_size = latent_size
        self._num_layers = num_mlp_layers
        self._mp_steps = num_message_passing_steps
        self._num_particle_types = num_particle_types

        self._embedding = hk.Embed(
            num_particle_types, particle_type_embedding_size
        )  # (9, 16)

    def _encoder(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """MLP graph encoder"""
        node_latents = build_mlp(
            self._latent_size, self._latent_size, self._num_layers
        )(graph.nodes)
        edge_latents = build_mlp(
            self._latent_size, self._latent_size, self._num_layers
        )(graph.edges)
        return jraph.GraphsTuple(
            nodes=node_latents,
            edges=edge_latents,
            globals=graph.globals,
            receivers=graph.receivers,
            senders=graph.senders,
            n_node=jnp.asarray([node_latents.shape[0]]),
            n_edge=jnp.asarray([edge_latents.shape[0]]),
        )

    def _processor(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """Sequence of 15 Graph Network blocks"""

        def update_edge_features(
            edge_features, sender_node_features, receiver_node_features, _  # globals_
        ):
            update_fn = build_mlp(
                self._latent_size, self._latent_size, self._num_layers
            )
            # Calculate sender node features from edge features
            return update_fn(
                jnp.concatenate(
                    [sender_node_features, receiver_node_features, edge_features],
                    axis=-1,
                )
            )

        def update_node_features(
            node_features,
            _,  # aggr_sender_edge_features,
            aggr_receiver_edge_features,
            __,  # globals_,
        ):
            update_fn = build_mlp(
                self._latent_size, self._latent_size, self._num_layers
            )
            features = [node_features, aggr_receiver_edge_features]
            return update_fn(jnp.concatenate(features, axis=-1))

        # Perform iterative message passing by stacking Graph Network blocks
        for _ in range(self._mp_steps):
            _graph = jraph.GraphNetwork(
                update_edge_fn=update_edge_features, update_node_fn=update_node_features
            )(graph)
            graph = graph._replace(
                nodes=_graph.nodes + graph.nodes, edges=_graph.edges + graph.edges
            )

        return graph

    def _decoder(self, graph: jraph.GraphsTuple):
        """MLP graph node decoder"""
        return build_mlp(
            self._latent_size, self._output_size, self._num_layers, is_layer_norm=False
        )(graph.nodes)

    def _transform(
        self, features: Dict[str, jnp.ndarray], particle_type: jnp.ndarray
    ) -> jraph.GraphsTuple:
        """Convert physical features to jraph.GraphsTuple for gns."""
        n_total_points = features["vel_hist"].shape[0]
        node_features = [
            features[k]
            for k in ["vel_hist", "vel_mag", "bound", "force"]
            if k in features
        ]
        edge_features = [features[k] for k in ["rel_disp", "rel_dist"] if k in features]

        graph = jraph.GraphsTuple(
            nodes=jnp.concatenate(node_features, axis=-1),
            edges=jnp.concatenate(edge_features, axis=-1),
            receivers=features["receivers"],
            senders=features["senders"],
            n_node=jnp.array([n_total_points]),
            n_edge=jnp.array([len(features["senders"])]),
            globals=None,
        )

        return graph, particle_type

    def __call__(
        self, sample: Tuple[Dict[str, jnp.ndarray], jnp.ndarray]
    ) -> jraph.GraphsTuple:
        graph, particle_type = self._transform(*sample)

        if self._num_particle_types > 1:
            particle_type_embeddings = self._embedding(particle_type)
            new_node_features = jnp.concatenate(
                [graph.nodes, particle_type_embeddings], axis=-1
            )
            graph = graph._replace(nodes=new_node_features)

        return self._decoder(self._processor(self._encoder(graph)))

    @classmethod
    def setup_model(cls, args: Namespace) -> Tuple["GNS", Type]:
        return cls(
            particle_dimension=args.metadata["dim"],
            latent_size=args.config.latent_dim,
            num_mlp_layers=args.config.num_mlp_layers,
            num_message_passing_steps=args.config.num_mp_steps,
            num_particle_types=NodeType.SIZE,
            particle_type_embedding_size=16,
        )
