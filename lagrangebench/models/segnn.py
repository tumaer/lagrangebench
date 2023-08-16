"""Steerable E(3) equivariant GNN. Model + feature transform, everything in one file."""
import warnings
from math import prod
from typing import Any, Callable, Dict, Optional, Tuple, Union

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp
import jraph
from e3nn_jax import Irreps, IrrepsArray
from e3nn_jax._src.tensor_products import naive_broadcast_decorator
from jax.tree_util import Partial, tree_map

from lagrangebench.utils import NodeType

from .base import BaseModel
from .utils import SteerableGraphsTuple, features_2d_to_3d


def uniform_init(
    name: str,
    path_shape: Tuple[int, ...],
    weight_std: float,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    return hk.get_parameter(
        name,
        shape=path_shape,
        dtype=dtype,
        init=hk.initializers.RandomUniform(minval=-weight_std, maxval=weight_std),
    )


class O3TensorProduct(hk.Module):
    """
    O(3) equivariant linear parametrized tensor product layer.

    Functionally the same as O3TensorProductLegacy, but around 5-10% faster.
    FullyConnectedTensorProduct seems faster than tensor_product + linear:
    https://github.com/e3nn/e3nn-jax/releases/tag/0.14.0
    """

    def __init__(
        self,
        output_irreps: e3nn.Irreps,
        *,
        biases: bool = True,
        name: Optional[str] = None,
        init_fn: Optional[Callable] = None,
        gradient_normalization: Optional[Union[str, float]] = "element",
        path_normalization: Optional[Union[str, float]] = "element",
    ):
        """Initialize the tensor product.

        Args:
            output_irreps: Output representation
            biases: If set ot true will add biases
            name: Name of the linear layer params
            init_fn: Weight initialization function. Default is uniform.
            gradient_normalization: Gradient normalization method.
            path_normalization: Path normalization method. Default is "element"
        """
        super().__init__(name)

        if not isinstance(output_irreps, e3nn.Irreps):
            output_irreps = e3nn.Irreps(output_irreps)
        self.output_irreps = output_irreps

        # tp weight init
        if not init_fn:
            init_fn = uniform_init
        self.get_parameter = init_fn

        self._gradient_normalization = gradient_normalization
        self._path_normalization = path_normalization

        self.biases = biases and "0e" in self.output_irreps

    def _build_tensor_product(
        self, left_irreps: e3nn.Irreps, right_irreps: e3nn.Irreps
    ) -> Callable:
        """Build the tensor product function."""
        tp = e3nn.FunctionalFullyConnectedTensorProduct(
            left_irreps,
            right_irreps,
            self.output_irreps,
            gradient_normalization=self._gradient_normalization,
            path_normalization=self._path_normalization,
        )
        ws = [
            self.get_parameter(
                name=(
                    f"w[{ins.i_in1},{ins.i_in2},{ins.i_out}] "
                    f"{tp.irreps_in1[ins.i_in1]},"
                    f"{tp.irreps_in2[ins.i_in2]},"
                    f"{tp.irreps_out[ins.i_out]}"
                ),
                path_shape=ins.path_shape,
                weight_std=ins.weight_std,
            )
            for ins in tp.instructions
        ]

        def tensor_product(x, y, **kwargs):
            return tp.left_right(ws, x, y, **kwargs)._convert(self.output_irreps)

        return naive_broadcast_decorator(tensor_product)

    def _build_biases(self) -> Callable:
        """Build the add bias function."""
        b = [
            self.get_parameter(
                f"b[{i_out}] {self.output_irreps}",
                path_shape=(mul_ir.dim,),
                weight_std=1 / jnp.sqrt(mul_ir.dim),
            )
            for i_out, mul_ir in enumerate(self.output_irreps)
            if mul_ir.ir.is_scalar()
        ]
        b = e3nn.IrrepsArray(f"{self.output_irreps.count('0e')}x0e", jnp.concatenate(b))

        # TODO: could be improved
        def _wrapper(x: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
            scalars = x.filter("0e")
            other = x.filter(drop="0e")
            return e3nn.concatenate(
                [scalars + b.broadcast_to(scalars.shape), other], axis=1
            )

        return _wrapper

    def __call__(
        self, x: e3nn.IrrepsArray, y: Optional[e3nn.IrrepsArray] = None, **kwargs
    ) -> e3nn.IrrepsArray:
        """Apply an O(3) equivariant linear parametrized tensor product layer.

        Args:
            x (IrrepsArray): Left tensor
            y (IrrepsArray): Right tensor. If None it defaults to np.ones.

        Returns:
            The output to the weighted tensor product (IrrepsArray).
        """
        if not y:
            y = e3nn.IrrepsArray("1x0e", jnp.ones((1, 1), dtype=x.dtype))

        if x.irreps.lmax == 0 and y.irreps.lmax == 0 and self.output_irreps.lmax > 0:
            warnings.warn(
                f"The specified output irreps ({self.output_irreps}) are not scalars "
                "but both operands are. This can have undesired behaviour (NaN). Try "
                "redistributing them into scalars or choose higher orders."
            )

        tp = self._build_tensor_product(x.irreps, y.irreps)
        output = tp(x, y, **kwargs)

        if self.biases:
            # add biases
            bias_fn = self._build_biases()
            return bias_fn(output)

        return output


def O3TensorProductGate(
    output_irreps: e3nn.Irreps,
    *,
    biases: bool = True,
    scalar_activation: Optional[Callable] = None,
    gate_activation: Optional[Callable] = None,
    name: Optional[str] = None,
    init_fn: Optional[Callable] = None,
) -> Callable:
    """Non-linear (gated) O(3) equivariant linear tensor product layer.

    The tensor product lifts the input representation to have gating scalars.

    Args:
        output_irreps: Output representation
        biases: Add biases
        scalar_activation: Activation function for scalars
        gate_activation: Activation function for higher order
        name: Name of the linear layer params

    Returns:
        Function that applies the gated tensor product layer.
    """
    if not isinstance(output_irreps, e3nn.Irreps):
        output_irreps = e3nn.Irreps(output_irreps)

    # lift output with gating scalars
    gate_irreps = e3nn.Irreps(
        f"{output_irreps.num_irreps - output_irreps.count('0e')}x0e"
    )
    tensor_product = O3TensorProduct(
        (gate_irreps + output_irreps).regroup(),
        biases=biases,
        name=name,
        init_fn=init_fn,
    )
    if not scalar_activation:
        scalar_activation = jax.nn.silu
    if not gate_activation:
        gate_activation = jax.nn.sigmoid

    def _gated_tensor_product(
        x: e3nn.IrrepsArray, y: Optional[e3nn.IrrepsArray] = None, **kwargs
    ) -> e3nn.IrrepsArray:
        tp = tensor_product(x, y, **kwargs)
        return e3nn.gate(tp, even_act=scalar_activation, odd_gate_act=gate_activation)

    return _gated_tensor_product


def O3Embedding(embed_irreps: Irreps, embed_edges: bool = True) -> Callable:
    """Linear steerable embedding.

    Embeds the graph nodes in the representation space :param embed_irreps:.

    Args:
        embed_irreps: Output representation
        embed_edges: If true also embed edges/message passing features

    Returns:
        Function to embed graph nodes (and optionally edges)
    """

    def _embedding(
        st_graph: SteerableGraphsTuple,
    ) -> SteerableGraphsTuple:
        graph = st_graph.graph
        nodes = O3TensorProduct(
            embed_irreps,
            name="embedding_nodes",
        )(graph.nodes, st_graph.node_attributes)
        st_graph = st_graph._replace(graph=graph._replace(nodes=nodes))

        # NOTE edge embedding is not in the original paper but can get good results
        if embed_edges:
            additional_message_features = O3TensorProduct(
                embed_irreps, name="embedding_msg_features"
            )
            (st_graph.additional_message_features, st_graph.edge_attributes)
            st_graph = st_graph._replace(
                additional_message_features=additional_message_features
            )

        return st_graph

    return _embedding


def O3Decoder(
    latent_irreps: Irreps,
    output_irreps: Irreps,
    blocks: int = 1,
):
    """Steerable decoder.

    Args:
        latent_irreps: Representation from the previous block
        output_irreps: Output representation
        blocks: Number of tensor product blocks in the decoder

    Returns:
        Decoded latent feature space to output space.
    """

    def _decoder(st_graph: SteerableGraphsTuple):
        nodes = st_graph.graph.nodes
        for i in range(blocks):
            nodes = O3TensorProductGate(latent_irreps, name=f"readout_{i}")(
                nodes, st_graph.node_attributes
            )

        return O3TensorProduct(output_irreps, name="output")(
            nodes, st_graph.node_attributes
        )

    return _decoder


class SEGNNLayer(hk.Module):
    """
    Steerable E(3) equivariant layer [#segnn].

    Applies a message passing step (GN) with equivariant message and update functions.
    """

    def __init__(
        self,
        output_irreps: Irreps,
        layer_num: int,
        blocks: int = 2,
        norm: Optional[str] = None,
        aggregate_fn: Optional[Callable] = jraph.segment_sum,
    ):
        """
        Initialize the layer.

        Args:
            output_irreps: Layer output representation
            layer_num: Numbering of the layer
            blocks: Number of tensor product blocks in the layer
            norm: Normalization type. Either be None, 'instance' or 'batch'
            aggregate_fn: Message aggregation function. Defaults to sum.
        """
        super().__init__(f"layer_{layer_num}")
        assert norm in ["batch", "instance", "none", None], f"Unknown norm '{norm}'"
        self._output_irreps = output_irreps
        self._blocks = blocks
        self._norm = norm
        self._aggregate_fn = aggregate_fn

    def _message(
        self,
        edge_attribute: IrrepsArray,
        additional_message_features: IrrepsArray,
        edge_features: Any,
        incoming: IrrepsArray,
        outgoing: IrrepsArray,
        globals_: Any,
    ) -> IrrepsArray:
        """Steerable equivariant message function."""
        _ = globals_
        _ = edge_features
        # create messages
        msg = e3nn.concatenate([incoming, outgoing], axis=-1)
        if additional_message_features is not None:
            msg = e3nn.concatenate([msg, additional_message_features], axis=-1)
        # message mlp (phi_m in the paper) steered by edge attributeibutes
        for i in range(self._blocks):
            msg = O3TensorProductGate(self._output_irreps, name=f"tp_{i}")(
                msg, edge_attribute
            )
        # NOTE: original implementation only applied batch norm to messages
        if self._norm == "batch":
            msg = e3nn.haiku.BatchNorm(irreps=self._output_irreps)(msg)
        return msg

    def _update(
        self,
        node_attribute: IrrepsArray,
        nodes: IrrepsArray,
        senders: Any,
        msg: IrrepsArray,
        globals_: Any,
    ) -> IrrepsArray:
        """Steerable equivariant update function."""
        _ = senders
        _ = globals_
        x = e3nn.concatenate([nodes, msg], axis=-1)
        # update mlp (phi_f in the paper) steered by node attributeibutes
        for i in range(self._blocks - 1):
            x = O3TensorProductGate(self._output_irreps, name=f"tp_{i}")(
                x, node_attribute
            )
        # last update layer without activation
        update = O3TensorProduct(self._output_irreps, name=f"tp_{self._blocks - 1}")(
            x, node_attribute
        )
        # residual connection
        nodes += update
        # message norm
        if self._norm in ["batch", "instance"]:
            nodes = e3nn.haiku.BatchNorm(
                irreps=self._output_irreps,
                instance=(self._norm == "instance"),
            )(nodes)
        return nodes

    def __call__(self, st_graph: SteerableGraphsTuple) -> SteerableGraphsTuple:
        """Perform a message passing step.

        Args:
            st_graph: Input graph

        Returns:
            The updated graph
        """
        # NOTE node_attributes, edge_attributes and additional_message_features
        #  are never updated within the message passing layers
        return st_graph._replace(
            graph=jraph.GraphNetwork(
                update_node_fn=Partial(self._update, st_graph.node_attributes),
                update_edge_fn=Partial(
                    self._message,
                    st_graph.edge_attributes,
                    st_graph.additional_message_features,
                ),
                aggregate_edges_for_nodes_fn=self._aggregate_fn,
            )(st_graph.graph)
        )


def weight_balanced_irreps(
    scalar_units: int, irreps_right: Irreps, lmax: int = None
) -> Irreps:
    """
    Determine left irreps so that the tensor product with irreps_right has at least
    scalar_units weights.

    Args:
        scalar_units: Number of weights
        irreps_right: Right irreps
        lmax: Maximum L of the left irreps

    Returns:
        Left irreps
    """
    # irrep order
    if lmax is None:
        lmax = irreps_right.lmax
    # linear layer with squdare weight matrix
    linear_weights = scalar_units**2
    # raise hidden features until enough weigths
    n = 0
    while True:
        n += 1
        irreps_left = (Irreps.spherical_harmonics(lmax) * n).sort().irreps.simplify()
        # number of paths
        tp_weights = sum(
            prod([irreps_left[i_1].mul ** 2, irreps_right[i_2].mul])
            for i_1, (_, ir_1) in enumerate(irreps_left)
            for i_2, (_, ir_2) in enumerate(irreps_right)
            for _, (_, ir_out) in enumerate(irreps_left)
            if ir_out in ir_1 * ir_2
        )
        if tp_weights >= linear_weights:
            break
    return Irreps(irreps_left)


class SEGNN(BaseModel):
    """Steerable E(3) equivariant network [#segnn].

    References:
        [#segnn] Brandstetter, Hesselink, van der Pol, Bekkers, Welling
        Geometric and Physical Quantities improve {E(3)} Equivariant Message Passing.
        ICLR 2021, https://arxiv.org/abs/2110.02905
    """

    def __init__(
        self,
        node_features_irreps: Irreps,
        edge_features_irreps: Irreps,
        scalar_units: int,
        lmax_hidden: int,
        lmax_attributes: int,
        output_irreps: Irreps,
        num_layers: int,
        n_vels: int,
        velocity_aggregate: str = "avg",
        homogeneous_particles: bool = True,
        norm: Optional[str] = None,
        blocks_per_layer: int = 2,
        embed_msg_features: bool = False,
    ):
        """
        Initialize the network.

        Args:
            node_features_irreps: Irreps of the node features.
            edge_features_irreps: Irreps of the additional message passing features.
            scalar_units: Hidden units (lower bound). Actual number depends on lmax.
            lmax_hidden: Maximum L of the hidden layer representations.
            lmax_attributes: Maximum L of the attributes.
            output_irreps: Output representation.
            num_layers: Number of message passing layers
            n_vels: Number of velocities in the history.
            velocity_aggregate: Velocity sequence aggregation method.
            homogeneous_particles: If all particles are of homogeneous type.
            norm: Normalization type. Either None, 'instance' or 'batch'
            blocks_per_layer: Number of tensor product blocks in each message passing
            embed_msg_features: Set to true to also embed edges/message passing features
        """
        super().__init__()

        # network
        self._attribute_irreps = Irreps.spherical_harmonics(lmax_attributes)
        self._hidden_irreps = weight_balanced_irreps(
            scalar_units, self._attribute_irreps, lmax_hidden
        )
        self._output_irreps = output_irreps
        self._num_layers = num_layers
        self._embed_msg_features = embed_msg_features
        self._norm = norm
        self._blocks_per_layer = blocks_per_layer

        self._embedding = O3Embedding(
            self._hidden_irreps,
            embed_edges=self._embed_msg_features,
        )

        self._decoder = O3Decoder(
            latent_irreps=self._hidden_irreps,
            output_irreps=output_irreps,
            blocks=self._blocks_per_layer,
        )

        # transform
        assert velocity_aggregate in [
            "avg",
            "last",
        ], "Invalid velocity aggregate. Must be one of 'avg', 'sum' or 'last'."
        self._node_features_irreps = node_features_irreps
        self._edge_features_irreps = edge_features_irreps
        self._velocity_aggregate = velocity_aggregate
        self._n_vels = n_vels
        self._homogeneous_particles = homogeneous_particles

    def _transform(
        self, features: Dict[str, jnp.ndarray], particle_type: jnp.ndarray
    ) -> Tuple[SteerableGraphsTuple, int]:
        """Convert physical features to SteerableGraphsTuple for segnn."""
        dim = features["vel_hist"].shape[1] // self._n_vels
        assert (
            dim == 3 or dim == 2
        ), "The velocity history should be of shape (n_nodes, n_vels * 3)."

        n_nodes = features["vel_hist"].shape[0]

        features["vel_hist"] = features["vel_hist"].reshape(n_nodes, self._n_vels, dim)

        if dim == 2:
            # add zeros for z component for E(3) equivariance
            features = features_2d_to_3d(features)

        if self._n_vels == 1:
            vel = jnp.squeeze(features["vel_hist"])
        else:
            if self._velocity_aggregate == "avg":
                vel = jnp.mean(features["vel_hist"], 1)
            if self._velocity_aggregate == "last":
                vel = features["vel_hist"][:, -1, :]

        rel_pos = features["rel_disp"]
        edge_attributes = e3nn.spherical_harmonics(
            self._attribute_irreps, rel_pos, normalize=True, normalization="integral"
        )
        vel_embedding = e3nn.spherical_harmonics(
            self._attribute_irreps, vel, normalize=True, normalization="integral"
        )
        # scatter edge attributes to nodes (density)
        node_attributes = vel_embedding
        scattered_edges = tree_map(
            lambda e: jraph.segment_mean(e, features["receivers"], n_nodes),
            edge_attributes,
        )
        node_attributes += scattered_edges
        # scalar attribute to 1 by default
        node_attributes.array = node_attributes.array.at[..., 0].set(1.0)

        node_features = [features["vel_hist"].reshape(n_nodes, self._n_vels * 3)]
        node_features += [
            features[k] for k in ["vel_mag", "bound", "force"] if k in features
        ]
        node_features = jnp.concatenate(node_features, axis=-1)

        if not self._homogeneous_particles:
            particles = jax.nn.one_hot(particle_type, NodeType.SIZE)
            node_features = jnp.concatenate([node_features, particles], axis=-1)

        edge_features = [features[k] for k in ["rel_disp", "rel_dist"] if k in features]
        edge_features = jnp.concatenate(edge_features, axis=-1)

        feature_graph = jraph.GraphsTuple(
            nodes=IrrepsArray(self._node_features_irreps, node_features),
            edges=None,
            senders=features["senders"],
            receivers=features["receivers"],
            n_node=jnp.array([n_nodes]),
            n_edge=jnp.array([len(features["senders"])]),
            globals=None,
        )
        st_graph = SteerableGraphsTuple(
            graph=feature_graph,
            node_attributes=node_attributes,
            edge_attributes=edge_attributes,
            additional_message_features=IrrepsArray(
                self._edge_features_irreps, edge_features
            ),
        )

        return st_graph, dim

    def _postprocess(self, nodes: IrrepsArray, dim: int) -> Dict[str, jnp.ndarray]:
        acc = jnp.squeeze(nodes.array)
        if dim == 2:
            acc = acc[:, :2]
        return {"acc": acc}

    def __call__(
        self, sample: Tuple[Dict[str, jnp.ndarray], jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        # feature transformation
        st_graph, dim = self._transform(*sample)
        # node (and edge) embedding
        st_graph = self._embedding(st_graph)
        # message passing
        for n in range(self._num_layers):
            st_graph = SEGNNLayer(
                self._hidden_irreps, n, blocks=self._blocks_per_layer, norm=self._norm
            )(st_graph)
        # readout
        nodes = self._decoder(st_graph)
        out = self._postprocess(nodes, dim)
        return out
