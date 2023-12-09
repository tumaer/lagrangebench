"""
Graph Network-based Simulator.
GNS model and feature transform.
"""

from typing import Dict, Tuple

import haiku as hk
import jax.numpy as jnp
import jraph

from lagrangebench.utils import NodeType

from .base import BaseModel
from .utils import build_mlp


class GNS(BaseModel):
    r"""Graph Network-based Simulator by
    `Sanchez-Gonzalez et al. <https://arxiv.org/abs/2002.09405>`_.

    GNS is the simples graph neural network applied to particle dynamics. It is built on
    the usual Graph Network architecture, with an encoder, a processor, and a decoder.

    .. math::
        \begin{align}
            \mathbf{m}_{ij}^{(t+1)} &= \phi \left(
                \mathbf{m}_{ij}^{(t)}, \mathbf{h}_i^{(t)}, \mathbf{h}_j^{(t)} \right) \\
            \mathbf{h}_i^{(t+1)} &= \psi \left(
                \mathbf{h}_i^{(t)}, \sum_{j \in \mathcal{N}(i)} \mathbf{m}_{ij}^{(t+1)}
                \right) \\
        \end{align}
    """

    def __init__(
        self,
        particle_dimension: int,
        latent_size: int,
        blocks_per_step: int,
        num_mp_steps: int,
        particle_type_embedding_size: int,
        num_particle_types: int = NodeType.SIZE,
    ):
        """Initialize the model.

        Args:
            particle_dimension: Space dimensionality (e.g. 2 or 3).
            latent_size: Size of the latent representations.
            blocks_per_step: Number of MLP layers per block. #A 'block' can be encoder, procesor or decoder
            num_mp_steps: Number of message passing steps.
            particle_type_embedding_size: Size of the particle type embedding.
            num_particle_types: Max number of particle types.
        """
        super().__init__()
        self._output_size = particle_dimension
        self._latent_size = latent_size
        self._blocks_per_step = blocks_per_step
        self._mp_steps = num_mp_steps
        self._num_particle_types = num_particle_types

        self._embedding = hk.Embed(
            num_particle_types, particle_type_embedding_size
        )  # (9, 16)    #num_particle_types=9 (see largrangebench-->utils.py) and particle_type_embedding_size=16 (default)

 


    def _encoder(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """MLP graph encoder."""
        #node_latents and edge_latents are arrays of sizes [num_nodes, latent_size] and [num_edges, latent_size]
        #input graph has nodes = (3200,28) and edges = (26485,3)
        node_latents = build_mlp(self._latent_size, self._latent_size, self._blocks_per_step)(graph.nodes)       #latent_size=128, blocks_per_step=num_mp_steps=2 
                             #inside build_mlp() : MLP structure: [128,128] which is the shape of the first hidden layer (128 nodes) and the output layer (128 nodes) of the 'encoder'
                             #when graph.nodes is passed as an input, it created the input layer of shape 28 and this is batched for 3200 nodes (particles) 
                             #so the final MLP is of the shape: [28,128,128] and node_latents 'per node' is of the shape 128 and for all the particles it is (3200,128)
                             #For 2D_RPF_3200_20kevery100 : 
                             #node_latents.shape = (3200,128) and edge_latents.shape = (26485,128)
                             #graph.nodes.shape =(3200,28)  [see output from __call__() at the bottom (3200,12)+(3200,16) --> (3200,28)]
                             #graph.edges.shape =(26485,3)  
        edge_latents = build_mlp(self._latent_size, self._latent_size, self._blocks_per_step)(graph.edges)
        #return another graph
        return jraph.GraphsTuple(
            nodes=node_latents, #(3200,128)
            edges=edge_latents, #(26485,128)
            globals=graph.globals,  #none for RPF_2D
            receivers=graph.receivers, #keep the senders and receivers the same as the old graph
            senders=graph.senders,
            n_node=jnp.asarray([node_latents.shape[0]]), #3200
            n_edge=jnp.asarray([edge_latents.shape[0]]), #26485
        )

    def _processor(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """Sequence of Graph Network blocks."""

        def update_edge_features(edge_features, sender_node_features, receiver_node_features, _ ): # globals_
            update_fn = build_mlp(self._latent_size, self._latent_size, self._blocks_per_step)
            # Calculate sender node features from edge features
            return update_fn(jnp.concatenate([sender_node_features, receiver_node_features, edge_features],axis=-1,))

        def update_node_features(
            node_features,              #(3200,128)
            _,  # aggr_sender_edge_features,
            aggr_receiver_edge_features,#(3200,128) $e_i'$
            __,  # globals_,
            ):
            update_fn = build_mlp(self._latent_size, self._latent_size, self._blocks_per_step)
            features = [node_features, aggr_receiver_edge_features]
            return update_fn(jnp.concatenate(features, axis=-1))

        # Perform iterative message passing by stacking Graph Network blocks
        for _ in range(self._mp_steps): #self._mp_steps=10
            _graph = jraph.GraphNetwork(update_edge_fn=update_edge_features, update_node_fn=update_node_features)(graph)
            graph = graph._replace(nodes=_graph.nodes + graph.nodes, edges=_graph.edges + graph.edges) # the graph size remains the same. only the node and edge features are updated
                                                                                                       #Addition for 'resnet'
        return graph

    def _decoder(self, graph: jraph.GraphsTuple):
        """MLP graph node decoder."""
        #MLP structure: [128,2] which is the hidden layer and the output layer of the 'decoder'
        return build_mlp( self._latent_size, self._output_size, self._blocks_per_step, is_layer_norm=False,)(graph.nodes)  #returns (3200,2)

    def _transform(  #Preprocessing step for encoder
        self, features: Dict[str, jnp.ndarray], particle_type: jnp.ndarray
    ) -> jraph.GraphsTuple:
        """Convert physical features to jraph.GraphsTuple for gns."""
        n_total_points = features["vel_hist"].shape[0]  #"bound" does not exist for RPF
        
        #For RPF_2D_3200_20k_every_100:
        #node_features contain only velocity history (3200,10) [10 is because it is flattened (5,2)] and external force (3200,2).
        
        #embedded_k = self._embedding(features["k"])  #embedded_k: (3200,16), we pass sample[1], because it contains the number of particles in the batch, which is 3200 for RPF_2D_3200_20k_every_100
        
        #features['embedded_k'] = embedded_k #add embedded_k to features dictionary
        
        node_features = [
            features[k]
            for k in ["vel_hist", "vel_mag", "bound", "force"]  #5 previous velocities + 1 vel magnitude + 1 distance_to_obstacle + 1 external external_force = 
            if k in features
        ]
        edge_features = [features[k] for k in ["rel_disp", "rel_dist"] if k in features] # 
        #create a graph using the node_features and edge_features and feed it to the encoder after adding additional node features
        graph = jraph.GraphsTuple(
            nodes=jnp.concatenate(node_features, axis=-1), # (3200,10) concatenated with (3200,2) --> (3200,12)
            edges=jnp.concatenate(edge_features, axis=-1), # (26485,2) concatenated with (26485,1) --> (26485,3)
            receivers=features["receivers"],
            senders=features["senders"],
            n_node=jnp.array([n_total_points]),
            n_edge=jnp.array([len(features["senders"])]),
            globals=None,
        )

        return graph, particle_type

    def __call__(
        self, sample: Tuple[Dict[str, jnp.ndarray], jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        graph, particle_type = self._transform(*sample)  
        #For RPF2D : graph.nodes.shape =(3200,12) and graph.edges.shape=(26485,3)
        #particle_type = (3200,) all are 0's as in RPF there are no other particle types
        #graph.nodes = (3200,12)
        if self._num_particle_types > 1: #adding new node features
            particle_type_embeddings = self._embedding(particle_type)  #particle_type_embeddings (3200,16)
            new_node_features = jnp.concatenate(
                [graph.nodes, particle_type_embeddings], axis=-1
            ) 
            graph = graph._replace(nodes=new_node_features) #new node features = (3200,12) + (3200,16) --> (3200,28)
        acc = self._decoder(self._processor(self._encoder(graph)))
        return {"acc": acc} #predicted acc of shape: (3200,2)
