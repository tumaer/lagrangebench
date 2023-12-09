from typing import Dict, Tuple

import haiku as hk
import jax.numpy as jnp
import jraph

from lagrangebench.utils import NodeType

from .base import BaseModel
from .utils import build_mlp


class PDE_Refiner(BaseModel):
    
    def __init__(
        self,
        problem_dimension: int,
        latent_size: int,
        number_of_layers: int,
        num_mp_steps: int,
    ):
        """Initialize the model.

        Args:
            problem_dimension: Space dimensionality (e.g. 2 or 3).
            latent_size: Size of the latent representations.
            number_of_layers: Number of MLP layers per block. #A 'block' can be encoder, procesor or decoder
            num_mp_steps: Number of message passing steps.
        """
        super().__init__()
        
        self._output_size = problem_dimension
        self._latent_size = latent_size
        self._number_of_layers = number_of_layers
        self._mp_steps = num_mp_steps
        
        self._embedding = hk.Embed(1, 16) 
        
        
    #Preprocessing step for encoder. Creates a graph from the input data
    def _transform(self, features: Dict[str, jnp.ndarray]) -> jraph.GraphsTuple:
        
        n_total_points = features["vel_hist"].shape[0] #3200 for RPF_2d
        
        features["embedded_k"]  = self._embedding(features["k"])  #embedded_k: (3200,16), we pass sample[1], because it contains the number of particles in the batch, which is 3200 for RPF_2D_3200_20k_every_100
        
        node_features = [
            features[k]
            for k in ["u_t_noised","vel_hist", "embedded_k"]  #1 previous velocity 
            if k in features
        ]
        edge_features = None
        
        graph = jraph.GraphsTuple(
        nodes=jnp.concatenate(node_features, axis=-1), 
        edges=edge_features, #no edge features
        receivers=features["receivers"],
        senders=features["senders"],
        n_node=jnp.array([n_total_points]),
        n_edge=jnp.array([len(features["senders"])]),
        globals=None,
        )
        
        return graph
    
    def _encoder(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        #node_latents is array of sizes [num_nodes, latent_size] for RPF 2D (3200,128), i.e. each particle has an associated 128 dimensional latent vector
        node_latents = build_mlp(self._latent_size, self._latent_size, self._number_of_layers)(graph.nodes)       #latent_size=128, blocks_per_step=num_mp_steps=2 

        #return another graph
        return jraph.GraphsTuple(
            nodes=node_latents, #(3200,128)
            edges=None, 
            globals=graph.globals,  #none for RPF_2D
            receivers=graph.receivers, 
            senders=graph.senders,
            n_node=jnp.asarray([node_latents.shape[0]]), #3200
            n_edge= graph.n_edge, 
        )
    
    def _processor(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:                
        
        def update_node_features(
            node_features,              #(3200,128)
            _,  # aggr_sender_edge_features,
            aggr_receiver_edge_features,#(3200,128) $e_i'$
            __,  # globals_,
            ):
            update_fn = build_mlp(self._latent_size, self._latent_size, self._number_of_layers)
            #features = [node_features, aggr_receiver_edge_features]
            return update_fn(node_features)      
        
        for _ in range(self._mp_steps): #self._mp_steps=10
            _graph = jraph.GraphNetwork(update_node_fn=update_node_features, update_edge_fn=None)(graph)
            graph = graph._replace(nodes=_graph.nodes + graph.nodes) #removed _graph.edges + graph.edges because there are no edge features
        return graph

    def _decoder(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        return build_mlp( self._latent_size, self._output_size, self._number_of_layers, is_layer_norm=False,)(graph.nodes)  #returns (3200,2)

    
    def __call__(self, sample: Tuple[Dict[str, jnp.ndarray], jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        graph = self._transform(sample[0])  #sample[0] is a dictionary of features, sample[1] is particle_type ,for RPF [0,0,.....0], 3200 times

        noise = self._decoder(self._processor(self._encoder(graph))) #noise.shape = (3200,2)
        return {"noise": noise} 