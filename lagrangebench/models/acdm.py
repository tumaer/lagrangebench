from typing import Dict, Tuple

import haiku as hk
import jax.numpy as jnp
import jraph

from lagrangebench.utils import NodeType, fourier_embedding

from .base import BaseModel
from .utils import build_mlp

class ACDM(BaseModel):
    def __init__(
        self,
        problem_dimension: int,
        latent_size: int,
        number_of_layers: int,
        num_mp_steps: int,
        particle_type_embedding_size: int,
        num_particle_types: int = NodeType.SIZE,
    ):
        """Initialize the model.

        Args:
            problem_dimension: Space dimensionality (e.g. 2 or 3).
            latent_size: Size of the latent representations.
            number_of_layers: Number of MLP layers per block.
            num_mp_steps: Number of message passing steps.
        """
        super().__init__()

        self._output_size = 6 #remove hardcoding later 
        self._latent_size = latent_size
        self._number_of_layers = number_of_layers
        self._mp_steps = num_mp_steps
        self._num_particle_types = num_particle_types

        self._embedding = hk.Embed(num_particle_types, particle_type_embedding_size)

    # Preprocessing step for encoder. Creates a graph from the input data
    def _transform(
        self, features: Dict[str, jnp.ndarray], particle_type: jnp.ndarray
    ) -> jraph.GraphsTuple:
        n_total_points = features["vel_hist"].shape[0]  # 3200 for RPF_2d

        features["embedded_k"] = fourier_embedding(features["k"], 64)
        node_features = [
            features[k]
            for k in [
                "noised_acc",
                "vel_hist",
                "embedded_k",
                "vel_mag",
                "bound",
                "force",
            ]  # 1 previous velocity
            if k in features
        ]

        edge_features = [features[k] for k in ["rel_disp", "rel_dist"] if k in features]

        graph = jraph.GraphsTuple(
            nodes=jnp.concatenate(node_features, axis=-1),
            edges=jnp.concatenate(edge_features, axis=-1),  # no edge features
            receivers=features["receivers"],
            senders=features["senders"],
            n_node=jnp.array([n_total_points]),
            n_edge=jnp.array([len(features["senders"])]),
            globals=None,
        )

        return graph, particle_type

    def _encoder(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        node_latents = build_mlp(
            self._latent_size, self._latent_size, self._number_of_layers
        )(graph.nodes)

        edge_latents = build_mlp(
            self._latent_size, self._latent_size, self._number_of_layers
        )(graph.edges)
        # return another graph
        return jraph.GraphsTuple(
            nodes=node_latents,  # (3200,128)
            edges=edge_latents,  # (26485,128)
            globals=graph.globals,  # none for RPF_2D
            receivers=graph.receivers,
            senders=graph.senders,
            n_node=jnp.asarray([node_latents.shape[0]]),  # 3200
            n_edge=jnp.asarray([edge_latents.shape[0]]),  # 26485
        )

    def _processor(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        def update_edge_features(
            edge_features, sender_node_features, receiver_node_features, _
        ):  # globals_
            update_fn = build_mlp(
                self._latent_size, self._latent_size, self._number_of_layers
            )
            # Calculate sender node features from edge features
            return update_fn(
                jnp.concatenate(
                    [sender_node_features, receiver_node_features, edge_features],
                    axis=-1,
                )
            )

        def update_node_features(
            node_features,  # (3200,128)
            _,  # aggr_sender_edge_features,
            aggr_receiver_edge_features,  # (3200,128) $e_i'$
            __,  # globals_,
        ):
            update_fn = build_mlp(
                self._latent_size, self._latent_size, self._number_of_layers
            )
            features = [node_features, aggr_receiver_edge_features]
            return update_fn(jnp.concatenate(features, axis=-1))

        for _ in range(self._mp_steps):  # self._mp_steps=10
            _graph = jraph.GraphNetwork(
                update_edge_fn=update_edge_features, update_node_fn=update_node_features
            )(graph)
            graph = graph._replace(
                nodes=_graph.nodes + graph.nodes, edges=_graph.edges + graph.edges
            )
            # Addition for 'resnet'
        return graph

    def _decoder(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        return build_mlp(
            self._latent_size,
            self._output_size,
            self._number_of_layers,
            is_layer_norm=False,
        )(graph.nodes)  # returns (3200,2)

    def __call__(
        self, sample: Tuple[Dict[str, jnp.ndarray], jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        graph, particle_type = self._transform(*sample)

        if self._num_particle_types > 1:  # adding new node features
            particle_type_embeddings = self._embedding(
                particle_type
            )  # particle_type_embeddings (3200,16)
            new_node_features = jnp.concatenate(
                [graph.nodes, particle_type_embeddings], axis=-1
            )
            graph = graph._replace(nodes=new_node_features)

        noise = self._decoder(
            self._processor(self._encoder(graph))
        )  # noise.shape = (3200,2)
        return {"noise": noise}
