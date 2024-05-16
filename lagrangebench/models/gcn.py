"""
Graph Convolutional Network (GCN) in flax.
Graph Convolution and feature transform.
"""

from typing import Dict, Tuple

import flax.linen as nn
import jax.numpy as jnp
import jraph

from lagrangebench.utils import NodeType

from .utils import FlaxMLP


class GCN(nn.Module):
    r"""Graph Convolutional Network by
    `Defferrard  et al. <https://arxiv.org/abs/1606.09375>`_ and
    `Kipf & Welling <https://arxiv.org/abs/1609.02907>`_.
    """

    output_size: int
    latent_size: int
    num_mp_steps: int
    blocks_per_step: int
    layers_pre_mp: int
    layers_post_mp: int
    particle_type_embedding_size: int
    residual: bool = True
    num_particle_types: int = NodeType.SIZE

    @nn.compact
    def __call__(
        self, sample: Tuple[Dict[str, jnp.ndarray], jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        graph, particle_type = self._transform(*sample)

        if self.num_particle_types > 1:
            # embed particle type
            particle_type_embeddings = nn.Embed(
                self.num_particle_types, self.particle_type_embedding_size
            )(particle_type)
            new_node_features = jnp.concatenate(
                [graph.nodes, particle_type_embeddings], axis=-1
            )
            graph = graph._replace(nodes=new_node_features)

        # encode
        nodes = FlaxMLP(
            latent_size=self.latent_size,
            output_size=self.latent_size,
            num_layers=self.layers_pre_mp,
        )(graph.nodes)
        graph = graph._replace(nodes=nodes)

        # Perform iterative message passing by stacking Graph Network blocks
        for _ in range(self.num_mp_steps):
            update_mlp = FlaxMLP(
                latent_size=self.latent_size,
                output_size=self.latent_size,
                num_layers=self.blocks_per_step,
            )
            nodes = jraph.GraphConvolution(update_node_fn=update_mlp)(graph).nodes
            if self.residual:
                nodes = nodes + graph.nodes
            graph = graph._replace(nodes=nodes)

        # readout
        acc = FlaxMLP(
            latent_size=self.latent_size,
            output_size=self.output_size,
            num_layers=self.layers_post_mp,
        )(graph.nodes)

        return {"acc": acc}

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

        graph = jraph.GraphsTuple(
            nodes=jnp.concatenate(node_features, axis=-1),
            edges=None,
            receivers=features["receivers"],
            senders=features["senders"],
            n_node=jnp.array([n_total_points]),
            n_edge=jnp.array([len(features["senders"])]),
            globals=None,
        )

        return graph, particle_type
