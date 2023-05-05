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

from equisph.simulate import NodeType

from .utils import SteerableGraphsTuple


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

    Attributes:
        left_irreps: Left input representation
        right_irreps: Right input representation
        output_irreps: Output representation
        get_parameter: Haiku parameter getter and init function
        tensor_product: Tensor product function
        biases: Bias wrapper function
    """

    def __init__(
        self,
        output_irreps: Irreps,
        *,
        left_irreps: Irreps,
        right_irreps: Optional[Irreps] = None,
        biases: bool = True,
        name: Optional[str] = None,
        init_fn: Optional[Callable] = None,
        gradient_normalization: Optional[Union[str, float]] = "element",
        path_normalization: Optional[Union[str, float]] = "element",
    ):
        """Initialize the tensor product.

        Args:
            output_irreps: Output representation
            left_irreps: Left input representation
            right_irreps: Right input representation (optional, defaults to 1x0e)
            biases: If set ot true will add biases
            name: Name of the linear layer params
            init_fn: Weight initialization function. Default is uniform.
            gradient_normalization: Gradient normalization method. Default is "path"
                NOTE: gradient_normalization="element" is default in torch and haiku.
            path_normalization: Path normalization method. Default is "element"
        """
        super().__init__(name)

        if not right_irreps:
            # NOTE: this is equivalent to a linear recombination of the left vectors
            right_irreps = Irreps("1x0e")

        if not isinstance(output_irreps, Irreps):
            output_irreps = Irreps(output_irreps)
        if not isinstance(left_irreps, Irreps):
            left_irreps = Irreps(left_irreps)
        if not isinstance(right_irreps, Irreps):
            right_irreps = Irreps(right_irreps)

        self.output_irreps = output_irreps

        self.right_irreps = right_irreps
        self.left_irreps = left_irreps

        if not init_fn:
            init_fn = uniform_init

        self.get_parameter = init_fn

        # NOTE FunctionalFullyConnectedTensorProduct appears to be faster than combining
        #  tensor_product+linear: https://github.com/e3nn/e3nn-jax/releases/tag/0.14.0
        #  Implementation adapted from e3nn.haiku.FullyConnectedTensorProduct
        tp = e3nn.FunctionalFullyConnectedTensorProduct(
            left_irreps,
            right_irreps,
            output_irreps,
            gradient_normalization=gradient_normalization,
            path_normalization=path_normalization,
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
            return tp.left_right(ws, x, y, **kwargs)._convert(output_irreps)

        self.tensor_product = naive_broadcast_decorator(tensor_product)
        self.biases = None

        if biases and "0e" in self.output_irreps:
            # add biases
            b = [
                self.get_parameter(
                    f"b[{i_out}] {tp.irreps_out[i_out]}",
                    path_shape=(mul_ir.dim,),
                    weight_std=1 / jnp.sqrt(mul_ir.dim),
                )
                for i_out, mul_ir in enumerate(output_irreps)
                if mul_ir.ir.is_scalar()
            ]
            b = IrrepsArray(f"{self.output_irreps.count('0e')}x0e", jnp.concatenate(b))

            # TODO: could be improved
            def _wrapper(x: IrrepsArray) -> IrrepsArray:
                scalars = x.filter("0e")
                other = x.filter(drop="0e")
                return e3nn.concatenate(
                    [scalars + b.broadcast_to(scalars.shape), other], axis=1
                )

            self.biases = _wrapper

    def __call__(
        self, x: IrrepsArray, y: Optional[IrrepsArray] = None, **kwargs
    ) -> IrrepsArray:
        """Applies an O(3) equivariant linear parametrized tensor product layer.

        Args:
            x (IrrepsArray): Left tensor
            y (IrrepsArray): Right tensor. If None it defaults to np.ones.

        Returns:
            The output to the weighted tensor product (IrrepsArray).
        """

        if not y:
            y = IrrepsArray("1x0e", jnp.ones((1, 1), dtype=x.dtype))

        if x.irreps.lmax == 0 and y.irreps.lmax == 0 and self.output_irreps.lmax > 0:
            warnings.warn(
                f"The specified output irreps ({self.output_irreps}) are not scalars "
                "but both operands are. This can have undesired behaviour (NaN). Try "
                "redistributing them into scalars or choose higher orders."
            )

        assert (
            x.irreps == self.left_irreps
        ), f"Left irreps do not match. Got {x.irreps}, expected {self.left_irreps}"
        assert (
            y.irreps == self.right_irreps
        ), f"Right irreps do not match. Got {y.irreps}, expected {self.right_irreps}"

        output = self.tensor_product(x, y, **kwargs)

        if self.biases:
            # add biases
            return self.biases(output)

        return output


def O3TensorProductGate(
    output_irreps: Irreps,
    *,
    left_irreps: Irreps,
    right_irreps: Optional[Irreps] = None,
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

    if not isinstance(output_irreps, Irreps):
        output_irreps = Irreps(output_irreps)

    # lift output with gating scalars
    gate_irreps = Irreps(f"{output_irreps.num_irreps - output_irreps.count('0e')}x0e")
    tensor_product = O3TensorProduct(
        (gate_irreps + output_irreps).regroup(),
        left_irreps=left_irreps,
        right_irreps=right_irreps,
        biases=biases,
        name=name,
        init_fn=init_fn,
    )
    if not scalar_activation:
        scalar_activation = jax.nn.silu
    if not gate_activation:
        gate_activation = jax.nn.sigmoid

    def _gated_tensor_product(
        x: IrrepsArray, y: Optional[IrrepsArray] = None, **kwargs
    ) -> IrrepsArray:
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
        # TODO update
        graph = st_graph.graph
        nodes = O3TensorProduct(
            embed_irreps,
            left_irreps=graph.nodes.irreps,
            right_irreps=st_graph.node_attributes.irreps,
            name="embedding_nodes",
        )(graph.nodes, st_graph.node_attributes)
        st_graph = st_graph._replace(graph=graph._replace(nodes=nodes))

        # NOTE edge embedding is not in the original paper but can get good results
        if embed_edges:
            additional_message_features = O3TensorProduct(
                embed_irreps,
                left_irreps=graph.nodes.irreps,
                right_irreps=st_graph.node_attributes.irreps,
                name="embedding_msg_features",
            )(
                st_graph.additional_message_features,
                st_graph.edge_attributes,
            )
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
            nodes = O3TensorProductGate(
                latent_irreps,
                left_irreps=nodes.irreps,
                right_irreps=st_graph.node_attributes.irreps,
                name=f"readout_{i}",
            )(nodes, st_graph.node_attributes)

        return O3TensorProduct(
            output_irreps,
            left_irreps=nodes.irreps,
            right_irreps=st_graph.node_attributes.irreps,
            name="output",
        )(nodes, st_graph.node_attributes)

    return _decoder


class SEGNNLayer(hk.Module):
    """
    Steerable E(3) equivariant layer.

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
            msg = O3TensorProductGate(
                self._output_irreps,
                left_irreps=msg.irreps,
                right_irreps=getattr(edge_attribute, "irreps", None),
                name=f"tp_{i}",
            )(msg, edge_attribute)
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
            x = O3TensorProductGate(
                self._output_irreps,
                left_irreps=x.irreps,
                right_irreps=getattr(node_attribute, "irreps", None),
                name=f"tp_{i}",
            )(x, node_attribute)
        # last update layer without activation
        update = O3TensorProduct(
            self._output_irreps,
            left_irreps=x.irreps,
            right_irreps=getattr(node_attribute, "irreps", None),
            name=f"tp_{self._blocks - 1}",
        )(x, node_attribute)
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
    Determines left Irreps such that the weighted tensor product left x right has
    (at least) scalar_units weights.
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


class SEGNN(hk.Module):
    """Steerable E(3) equivariant network.

    Original paper https://arxiv.org/abs/2110.02905.
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
        )

        # transform
        if self.__class__ == SEGNN:
            assert velocity_aggregate in [
                "avg",
                "sum",
                "last",
            ], "Invalid velocity aggregate. Must be one of 'avg', 'sum' or 'last'."
        self._node_features_irreps = node_features_irreps
        self._edge_features_irreps = edge_features_irreps
        self._velocity_aggregate = velocity_aggregate
        self._n_vels = n_vels
        self._homogeneous_particles = homogeneous_particles

    def _transform(
        self, features: Dict[str, jnp.ndarray], particle_type: jnp.ndarray
    ) -> SteerableGraphsTuple:
        """Convert physical features to SteerableGraphsTuple for segnn."""

        assert (
            features["vel_hist"].shape[1] // self._n_vels == 3
        ), "The velocity history should be of shape (n_nodes, n_vels * 3)."

        traj = jnp.reshape(
            features["vel_hist"], (features["vel_hist"].shape[0], self._n_vels, 3)
        )

        if self._n_vels == 1 or self._velocity_aggregate == "all":
            vel = jnp.squeeze(traj)
        else:
            if self._velocity_aggregate == "avg":
                vel = jnp.mean(traj, 1)
            if self._velocity_aggregate == "sum":
                vel = jnp.sum(traj, 1)
            if self._velocity_aggregate == "last":
                vel = traj[:, -1, :]

        rel_pos = features["rel_disp"]

        edge_attributes = e3nn.spherical_harmonics(
            self._attribute_irreps, rel_pos, normalize=True, normalization="integral"
        )
        vel_embedding = e3nn.spherical_harmonics(
            self._attribute_irreps, vel, normalize=True, normalization="integral"
        )
        # scatter edge attributes
        sum_n_node = features["vel_hist"].shape[0]
        node_attributes = vel_embedding
        scattered_edges = tree_map(
            lambda e: jraph.segment_mean(e, features["receivers"], sum_n_node),
            edge_attributes,
        )

        if self._velocity_aggregate == "all":
            # transpose for broadcasting
            node_attributes.array = jnp.transpose(
                (
                    jnp.transpose(node_attributes.array, (0, 2, 1))
                    + jnp.expand_dims(scattered_edges.array, -1)
                ),
                (0, 2, 1),
            )
        else:
            node_attributes += scattered_edges

        # scalar attribute to 1 by default
        node_attributes.array = node_attributes.array.at[..., 0].set(1.0)

        node_features = [
            features[k]
            for k in ["vel_hist", "vel_mag", "bound", "force"]
            if k in features
        ]
        node_features = jnp.concatenate(node_features, axis=-1)

        if not self._homogeneous_particles:
            particles = jax.nn.one_hot(particle_type, NodeType.SIZE)
            node_features = jnp.concatenate([node_features, particles], axis=-1)

        edge_features = [features[k] for k in ["rel_disp", "rel_dist"] if k in features]
        edge_features = jnp.concatenate(edge_features, axis=-1)

        return SteerableGraphsTuple(
            graph=jraph.GraphsTuple(
                nodes=IrrepsArray(self._node_features_irreps, node_features),
                edges=None,
                senders=features["senders"],
                receivers=features["receivers"],
                n_node=jnp.array([sum_n_node]),
                n_edge=jnp.array([len(features["senders"])]),
                globals=None,
            ),
            node_attributes=node_attributes,
            edge_attributes=edge_attributes,
            additional_message_features=IrrepsArray(
                self._edge_features_irreps, edge_features
            ),
        )

    def __call__(self, sample: Tuple[Dict[str, jnp.ndarray], jnp.ndarray]) -> jnp.array:
        # feature transformation
        st_graph = self._transform(*sample)
        # node (and edge) embedding
        st_graph = self._embedding(st_graph)
        # message passing
        for n in range(self._num_layers):
            st_graph = SEGNNLayer(self._hidden_irreps, n, norm=self._norm)(st_graph)
        # readout
        nodes = self._decoder(st_graph)
        return jnp.squeeze(nodes.array)
