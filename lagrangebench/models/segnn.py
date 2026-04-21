"""
Steerable E(3) equivariant GNN from
`Brandstetter et al. <https://arxiv.org/abs/2110.02905>`_.
SEGNN model, layers and feature transform.

Original implementation: https://github.com/RobDHess/Steerable-E3-GNN

Standalone implementation + validation: https://github.com/gerkone/segnn-jax
"""


import warnings
from math import prod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import e3nn_jax as e3nn
import equinox as eqx
import jax
import jax.numpy as jnp
import jraph
from e3nn_jax import Irreps, IrrepsArray
from e3nn_jax._src.linear import FunctionalLinear
from jax.tree_util import tree_map

from lagrangebench.utils import NodeType

from .base import BaseModel
from .utils import SteerableGraphsTuple, features_2d_to_3d


def _resolve_irreps_io(
    irreps_in: Irreps, irreps_out: Irreps, simplify: bool = True
) -> Tuple[Irreps, Irreps]:
    """Match ``e3nn.flax.Linear``'s irreps-in/out simplification and filtering."""
    irreps_in = Irreps(irreps_in)
    irreps_out = Irreps(irreps_out)
    if simplify:
        irreps_in = irreps_in.simplify()
        irreps_out = irreps_out.simplify()
    # only keep outputs that are reachable from the input
    irreps_out = irreps_out.filter(keep=irreps_in)
    return irreps_in, irreps_out


class O3TensorProduct(eqx.Module):
    r"""
    O(3) equivariant linear parametrized tensor product layer.

    Applies a linear (parametrized) tensor product of representations to the input(s).

    .. math::
        \begin{align}
            tp(x, y) := \mathbf{x} \otimes_{CG}^{\mathcal{W}} \mathbf{y}
        \end{align}

    where :math:`\mathcal{W}` are learnable parameters.

    Uses :code:`tensor_product` + :code:`Linear` instead of FullyConnectedTensorProduct.
    From e3nn 0.19.2 (https://github.com/e3nn/e3nn-jax/releases/tag/0.19.2), this is
    as fast as FullyConnectedTensorProduct.
    """

    weights: List[jnp.ndarray]
    _linear: FunctionalLinear = eqx.field(static=True)
    output_irreps: Irreps = eqx.field(static=True)

    def __init__(
        self,
        x_irreps: Irreps,
        output_irreps: Irreps,
        y_irreps: Optional[Irreps] = None,
        *,
        biases: bool = True,
        init_scale: float = 1.0,
        gradient_normalization: Union[str, float] = "element",
        path_normalization: Union[str, float] = "element",
        key,
    ):
        """Initialize the tensor product.

        Args:
            x_irreps: Irreps of the left input ``x``.
            output_irreps: Output representation
            y_irreps: Irreps of the right input ``y``. Defaults to the scalar
                ``1x0e`` placeholder used when ``y`` is omitted at call time.
            biases: If set ot true will add biases
            init_scale: Multiplicative factor applied to the per-path weight std
                when sampling the weights with a uniform distribution in
                ``[-weight_std, weight_std]``.
            gradient_normalization: Gradient normalization method. Default is "element".
            NOTE: gradient_normalization="element" is the default in torch and haiku.
            path_normalization: Path normalization method. Default is "element"
            key: PRNG key for initialization.
        """
        x_irreps = Irreps(x_irreps)
        if y_irreps is None:
            y_irreps = Irreps("1x0e")
        else:
            y_irreps = Irreps(y_irreps)
        output_irreps = Irreps(output_irreps)

        # the actual input to the internal linear layer is tensor_product(x, y)
        tp_irreps = e3nn.tensor_product(x_irreps, y_irreps)

        # check for unreachable outputs (warn as original code did)
        if x_irreps.lmax == 0 and y_irreps.lmax == 0 and output_irreps.lmax > 0:
            warnings.warn(
                f"The specified output irreps ({output_irreps}) are not scalars "
                "but both operands are. This can have undesired behaviour (NaN). Try "
                "redistributing them into scalars or choose higher orders."
            )
        miss = output_irreps.filter(drop=tp_irreps)
        if len(miss) > 0:
            warnings.warn(f"Output irreps: '{miss}' are unreachable and were ignored.")

        lin_in_irreps, lin_out_irreps = _resolve_irreps_io(tp_irreps, output_irreps)
        linear = FunctionalLinear(
            lin_in_irreps,
            lin_out_irreps,
            biases=(biases and "0e" in lin_out_irreps),
            gradient_normalization=gradient_normalization,
            path_normalization=path_normalization,
        )
        self._linear = linear
        self.output_irreps = lin_out_irreps

        keys = jax.random.split(key, max(len(linear.instructions), 1))
        weights = []
        for k, ins in zip(keys, linear.instructions):
            std = ins.weight_std * init_scale
            weights.append(
                jax.random.uniform(k, ins.path_shape, minval=-std, maxval=std)
            )
        self.weights = weights

    def __call__(
        self, x: IrrepsArray, y: Optional[IrrepsArray] = None
    ) -> IrrepsArray:
        """Applies an O(3) equivariant linear parametrized tensor product layer.

        Args:
            x (IrrepsArray): Left tensor
            y (IrrepsArray): Right tensor. If None it defaults to np.ones.

        Returns:
            The output to the weighted tensor product (IrrepsArray).
        """
        if y is None:
            y = e3nn.IrrepsArray("1x0e", jnp.ones((1, 1), dtype=x.dtype))
        tp = e3nn.tensor_product(x, y)
        tp = tp.remove_zero_chunks().regroup()

        f = lambda arr: self._linear(self.weights, arr)
        for _ in range(tp.ndim - 1):
            f = e3nn.utils.vmap(f)
        return f(tp)


class O3TensorProductGate(eqx.Module):
    r"""Non-linear (gated) O(3) equivariant linear tensor product layer.

    It applies a linear tensor product of representations to the input(s) and then
    a gated nonlinearity by `Weiler et al. <https://arxiv.org/abs/1807.02547>`_.

    The input representation is lifted to have gating scalars.

    Args:
        x_irreps: Irreps of the left input ``x``.
        output_irreps: Output representation
        y_irreps: Irreps of the right input ``y``.
        biases: Add biases
        scalar_activation: Activation function for scalars
        gate_activation: Activation function for higher order
        key: PRNG key for initialization.
    """

    tp: O3TensorProduct
    scalar_activation: Callable = eqx.field(static=True)
    gate_activation: Callable = eqx.field(static=True)

    def __init__(
        self,
        x_irreps: Irreps,
        output_irreps: Irreps,
        y_irreps: Optional[Irreps] = None,
        *,
        biases: bool = True,
        scalar_activation: Optional[Callable] = None,
        gate_activation: Optional[Callable] = None,
        init_scale: float = 1.0,
        key,
    ):
        output_irreps = Irreps(output_irreps)
        gate_irreps = Irreps(
            f"{output_irreps.num_irreps - output_irreps.count('0e')}x0e"
        )
        self.tp = O3TensorProduct(
            x_irreps,
            (gate_irreps + output_irreps).regroup(),
            y_irreps=y_irreps,
            biases=biases,
            init_scale=init_scale,
            key=key,
        )
        if scalar_activation is None:
            scalar_activation = jax.nn.silu
        if gate_activation is None:
            gate_activation = jax.nn.sigmoid
        self.scalar_activation = scalar_activation
        self.gate_activation = gate_activation

    def __call__(
        self, x: IrrepsArray, y: Optional[IrrepsArray] = None
    ) -> IrrepsArray:
        tp = self.tp(x, y)
        return e3nn.gate(
            tp,
            even_act=self.scalar_activation,
            odd_gate_act=self.gate_activation,
        )


class O3Embedding(eqx.Module):
    """Linear steerable embedding.

    Embeds the graph nodes in the representation space :param embed_irreps:.

    Args:
        embed_irreps: Output representation
        embed_edges: If true also embed edges/message passing features
    """

    node_embed: O3TensorProduct
    msg_embed: Optional[O3TensorProduct]
    embed_edges: bool = eqx.field(static=True)

    def __init__(
        self,
        node_in_irreps: Irreps,
        node_attr_irreps: Irreps,
        edge_in_irreps: Optional[Irreps],
        edge_attr_irreps: Optional[Irreps],
        embed_irreps: Irreps,
        embed_edges: bool = True,
        *,
        key,
    ):
        self.embed_edges = embed_edges
        k_nodes, k_msg = jax.random.split(key)
        self.node_embed = O3TensorProduct(
            node_in_irreps,
            embed_irreps,
            y_irreps=node_attr_irreps,
            key=k_nodes,
        )
        if embed_edges:
            assert edge_in_irreps is not None and edge_attr_irreps is not None
            self.msg_embed = O3TensorProduct(
                edge_in_irreps,
                embed_irreps,
                y_irreps=edge_attr_irreps,
                key=k_msg,
            )
        else:
            self.msg_embed = None

    def __call__(self, st_graph: SteerableGraphsTuple) -> SteerableGraphsTuple:
        graph = st_graph.graph
        nodes = self.node_embed(graph.nodes, st_graph.node_attributes)
        st_graph = st_graph._replace(graph=graph._replace(nodes=nodes))
        if self.embed_edges:
            amf = self.msg_embed(
                st_graph.additional_message_features, st_graph.edge_attributes
            )
            st_graph = st_graph._replace(additional_message_features=amf)
        return st_graph


class O3Decoder(eqx.Module):
    """Steerable decoder.

    Args:
        latent_irreps: Representation from the previous block
        output_irreps: Output representation
        n_blocks: Number of tensor product blocks in the decoder
    """

    readout_blocks: List[O3TensorProductGate]
    output_block: O3TensorProduct

    def __init__(
        self,
        latent_irreps: Irreps,
        node_attr_irreps: Irreps,
        output_irreps: Irreps,
        n_blocks: int = 1,
        *,
        key,
    ):
        keys = jax.random.split(key, n_blocks + 1)
        self.readout_blocks = [
            O3TensorProductGate(
                latent_irreps,
                latent_irreps,
                y_irreps=node_attr_irreps,
                key=keys[i],
            )
            for i in range(n_blocks)
        ]
        self.output_block = O3TensorProduct(
            latent_irreps,
            output_irreps,
            y_irreps=node_attr_irreps,
            key=keys[-1],
        )

    def __call__(self, st_graph: SteerableGraphsTuple) -> IrrepsArray:
        nodes = st_graph.graph.nodes
        for block in self.readout_blocks:
            nodes = block(nodes, st_graph.node_attributes)
        return self.output_block(nodes, st_graph.node_attributes)


class SEGNNLayer(eqx.Module):
    """
    Steerable E(3) equivariant layer.

    Applies a message passing step (GN) with equivariant message and update functions.
    """

    msg_blocks: List[O3TensorProductGate]
    update_gate_blocks: List[O3TensorProductGate]
    update_final: O3TensorProduct
    output_irreps: Irreps = eqx.field(static=True)
    n_blocks: int = eqx.field(static=True)
    aggregate_fn: Callable = eqx.field(static=True)

    def __init__(
        self,
        msg_in_irreps: Irreps,
        node_in_irreps: Irreps,
        edge_attr_irreps: Irreps,
        node_attr_irreps: Irreps,
        output_irreps: Irreps,
        n_blocks: int = 2,
        aggregate_fn: Callable = jraph.segment_sum,
        *,
        key,
    ):
        """
        Initialize the layer.

        Args:
            msg_in_irreps: Irreps of the pre-message concatenation
                ``[h_i, h_j, additional_message_features]`` at the layer input.
            node_in_irreps: Irreps of the node features at the layer input.
            edge_attr_irreps: Irreps of the steerable edge attributes.
            node_attr_irreps: Irreps of the steerable node attributes.
            output_irreps: Layer output representation
            n_blocks: Number of tensor product n_blocks in the layer
            aggregate_fn: Message aggregation function. Defaults to sum.
            key: PRNG key for initialization.
        """
        self.output_irreps = Irreps(output_irreps)
        self.n_blocks = n_blocks
        self.aggregate_fn = aggregate_fn

        output_irreps_clean = Irreps(output_irreps)

        key_m, key_u = jax.random.split(key, 2)
        keys_m = jax.random.split(key_m, n_blocks)
        self.msg_blocks = []
        cur_irreps = Irreps(msg_in_irreps)
        for i in range(n_blocks):
            block = O3TensorProductGate(
                cur_irreps,
                output_irreps_clean,
                y_irreps=edge_attr_irreps,
                key=keys_m[i],
            )
            self.msg_blocks.append(block)
            # after gate the irreps collapse back to ``output_irreps``
            cur_irreps = output_irreps_clean

        keys_u = jax.random.split(key_u, n_blocks)
        self.update_gate_blocks = []
        # update takes [nodes, msg] concatenated -> node_in_irreps + output_irreps
        cur_irreps = (Irreps(node_in_irreps) + output_irreps_clean).regroup()
        for i in range(n_blocks - 1):
            block = O3TensorProductGate(
                cur_irreps,
                output_irreps_clean,
                y_irreps=node_attr_irreps,
                key=keys_u[i],
            )
            self.update_gate_blocks.append(block)
            cur_irreps = output_irreps_clean
        self.update_final = O3TensorProduct(
            cur_irreps,
            output_irreps_clean,
            y_irreps=node_attr_irreps,
            key=keys_u[-1],
        )

    def _message(
        self,
        incoming: IrrepsArray,
        outgoing: IrrepsArray,
        additional_message_features: Optional[IrrepsArray],
        edge_attribute: IrrepsArray,
    ) -> IrrepsArray:
        """Steerable equivariant message function."""
        msg = e3nn.concatenate([incoming, outgoing], axis=-1)
        if additional_message_features is not None:
            msg = e3nn.concatenate([msg, additional_message_features], axis=-1)
        for block in self.msg_blocks:
            msg = block(msg, edge_attribute)
        return msg

    def _update(
        self,
        nodes: IrrepsArray,
        msg_sum: IrrepsArray,
        node_attribute: IrrepsArray,
    ) -> IrrepsArray:
        """Steerable equivariant update function."""
        x = e3nn.concatenate([nodes, msg_sum], axis=-1)
        for block in self.update_gate_blocks:
            x = block(x, node_attribute)
        update = self.update_final(x, node_attribute)
        return nodes + update

    def __call__(self, st_graph: SteerableGraphsTuple) -> SteerableGraphsTuple:
        """Perform a message passing step.

        Args:
            st_graph: Input graph

        Returns:
            The updated graph
        """
        graph = st_graph.graph
        n_nodes = graph.nodes.shape[0]

        incoming = tree_map(lambda a: a[graph.senders], graph.nodes)
        outgoing = tree_map(lambda a: a[graph.receivers], graph.nodes)
        msg = self._message(
            incoming,
            outgoing,
            st_graph.additional_message_features,
            st_graph.edge_attributes,
        )
        # aggregate messages to receivers; apply segment_sum per chunk
        msg_sum = tree_map(
            lambda a: self.aggregate_fn(a, graph.receivers, n_nodes), msg
        )
        new_nodes = self._update(graph.nodes, msg_sum, st_graph.node_attributes)
        return st_graph._replace(graph=graph._replace(nodes=new_nodes))


def weight_balanced_irreps(
    scalar_units: int, irreps_right: Irreps, lmax: Optional[int] = None
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
    r"""
    Steerable E(3) equivariant network by
    `Brandstetter et al. <https://arxiv.org/abs/2110.02905>`_.

    SEGNNs are E(3)-equivariant graph neural networks based around tensor products of
    representations. By design, SEGNNs allow for flexible scalar/vectorial inputs and
    outputs for both edges and attributes. The message passing is modified as follows:

    .. math::
        \begin{align}
            \mathbf{m}_{ij} &= \textit{M}_{\mathbf{\hat{a}}_{ij}}\left(
                \mathbf{f}_i, \mathbf{f}_j, \| x_i - x_j \|^2 \right), \\
            \mathbf{f}^{\prime}_i &= \textit{U}_{\mathbf{\hat{a}}_i}\left(
                \mathbf{f}_i, \sum_{j\in\mathcal{N}(i)} \mathbf{m}_{ij} \right)
        \end{align}

    :math:`\mathbf{\hat{a}}_{ij}` and :math:`\mathbf{\hat{a}}_{i}` are edge and
    node attributes and the operators :math:`\textit{M}_{\mathbf{\hat{a}}_{ij}}` and
    :math:`\textit{U}_{\mathbf{\hat{a}}_{i}}` are defined as a tensor product of
    representations :math:`\otimes_{CG}` between the input and the attribues:

    .. math::
        \begin{align}
            \textit{U}_{\mathbf{\hat{a}}_i} :=
            \mathcal{W^n}_{\mathbf{\hat{a}}_i}
            (\dots \sigma ( \mathcal{W^0}_{\mathbf{\hat{a}}_i} \mathbf{f} ) )
            \quad \text{with} \quad
            \mathcal{W}_{\mathbf{\hat{a}}_i} \mathbf{f} :=
            \mathbf{f} \otimes_{CG}^{\mathcal{W}} \mathbf{\mathbf{\hat{a}}}
        \end{align}

    where :math`\mathbf{f} = \[\mathbf{f}_i,\sum_{j\in\mathcal{N}(i)} \mathbf{m}_{ij}\]`
    are node features concatenated to the aggregated messages, :math:`\sigma` is a gated
    non-linearity and :math:`\mathcal{W}` are the  tensor product parameters.
    :math:`\textit{M}_{\mathbf{\hat{a}}_{ij}}` is similarly defined, but with the
    nonlinearity on the last layer, with edge attributes :math:`\mathbf{\hat{a}}_{ij}`
    and :math`\mathbf{f} = \[ \mathbf{f}_i, \mathbf{f}_j, \|x_i - x_j\|^2 \]`

    """

    embedding: O3Embedding
    layers: List[SEGNNLayer]
    decoder: O3Decoder

    attribute_irreps: Irreps = eqx.field(static=True)
    hidden_irreps: Irreps = eqx.field(static=True)
    output_irreps: Irreps = eqx.field(static=True)
    node_features_irreps: Irreps = eqx.field(static=True)
    edge_features_irreps: Irreps = eqx.field(static=True)
    num_mp_steps: int = eqx.field(static=True)
    blocks_per_step: int = eqx.field(static=True)
    embed_msg_features: bool = eqx.field(static=True)
    velocity_aggregate: str = eqx.field(static=True)
    n_vels: int = eqx.field(static=True)
    homogeneous_particles: bool = eqx.field(static=True)

    def __init__(
        self,
        node_features_irreps: Irreps,
        edge_features_irreps: Irreps,
        scalar_units: int,
        lmax_hidden: int,
        lmax_attributes: int,
        output_irreps: Irreps,
        num_mp_steps: int,
        n_vels: int,
        velocity_aggregate: str = "avg",
        homogeneous_particles: bool = True,
        blocks_per_step: int = 2,
        embed_msg_features: bool = False,
        *,
        key,
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
            num_mp_steps: Number of message passing layers
            n_vels: Number of velocities in the history.
            velocity_aggregate: Velocity sequence aggregation method.
            homogeneous_particles: If all particles are of homogeneous type.
            blocks_per_step: Number of tensor product blocks in each message passing
            embed_msg_features: Set to true to also embed edges/message passing features
            key: PRNG key for initialization.
        """
        self.attribute_irreps = Irreps.spherical_harmonics(lmax_attributes)
        self.hidden_irreps = weight_balanced_irreps(
            scalar_units, self.attribute_irreps, lmax_hidden
        )
        self.output_irreps = Irreps(output_irreps)
        self.node_features_irreps = Irreps(node_features_irreps)
        self.edge_features_irreps = Irreps(edge_features_irreps)
        self.num_mp_steps = num_mp_steps
        self.blocks_per_step = blocks_per_step
        self.embed_msg_features = embed_msg_features

        assert velocity_aggregate in [
            "avg",
            "last",
        ], "Invalid velocity aggregate. Must be one of 'avg', 'sum' or 'last'."
        self.velocity_aggregate = velocity_aggregate
        self.n_vels = n_vels
        self.homogeneous_particles = homogeneous_particles

        keys = jax.random.split(key, 2 + num_mp_steps)
        self.embedding = O3Embedding(
            node_in_irreps=self.node_features_irreps,
            node_attr_irreps=self.attribute_irreps,
            edge_in_irreps=self.edge_features_irreps if embed_msg_features else None,
            edge_attr_irreps=self.attribute_irreps if embed_msg_features else None,
            embed_irreps=self.hidden_irreps,
            embed_edges=embed_msg_features,
            key=keys[0],
        )

        # message passing layers
        self.layers = []
        # input to message block: [h_i, h_j, msg_features] concat
        msg_feat_irreps = (
            self.hidden_irreps if embed_msg_features else self.edge_features_irreps
        )
        msg_in_irreps = (
            (self.hidden_irreps + self.hidden_irreps + msg_feat_irreps)
            .regroup()
        )
        for i in range(num_mp_steps):
            self.layers.append(
                SEGNNLayer(
                    msg_in_irreps=msg_in_irreps,
                    node_in_irreps=self.hidden_irreps,
                    edge_attr_irreps=self.attribute_irreps,
                    node_attr_irreps=self.attribute_irreps,
                    output_irreps=self.hidden_irreps,
                    n_blocks=blocks_per_step,
                    key=keys[1 + i],
                )
            )

        self.decoder = O3Decoder(
            latent_irreps=self.hidden_irreps,
            node_attr_irreps=self.attribute_irreps,
            output_irreps=self.output_irreps,
            n_blocks=blocks_per_step,
            key=keys[-1],
        )

    def _transform(
        self, features: Dict[str, jnp.ndarray], particle_type: jnp.ndarray
    ) -> Tuple[SteerableGraphsTuple, int]:
        """Convert physical features to SteerableGraphsTuple for segnn."""
        dim = features["vel_hist"].shape[1] // self.n_vels
        assert (
            dim == 3 or dim == 2
        ), "The velocity history should be of shape (n_nodes, n_vels * 3)."

        n_nodes = features["vel_hist"].shape[0]
        features["vel_hist"] = features["vel_hist"].reshape(n_nodes, self.n_vels, dim)
        if dim == 2:
            features = features_2d_to_3d(features)

        if self.n_vels == 1:
            vel = jnp.squeeze(features["vel_hist"])
        else:
            if self.velocity_aggregate == "avg":
                vel = jnp.mean(features["vel_hist"], 1)
            if self.velocity_aggregate == "last":
                vel = features["vel_hist"][:, -1, :]

        rel_pos = features["rel_disp"]
        edge_attributes = e3nn.spherical_harmonics(
            self.attribute_irreps, rel_pos, normalize=True, normalization="integral"
        )
        vel_embedding = e3nn.spherical_harmonics(
            self.attribute_irreps, vel, normalize=True, normalization="integral"
        )
        scattered_edges = tree_map(
            lambda e: jraph.segment_mean(e, features["receivers"], n_nodes),
            edge_attributes,
        )
        node_attributes = e3nn.IrrepsArray(
            vel_embedding.irreps,
            (vel_embedding + scattered_edges).array.at[:, 0].set(1.0),
        )

        node_features = [features["vel_hist"].reshape(n_nodes, self.n_vels * 3)]
        node_features += [
            features[k] for k in ["vel_mag", "bound", "force"] if k in features
        ]
        node_features = jnp.concatenate(node_features, axis=-1)

        if not self.homogeneous_particles:
            particles = jax.nn.one_hot(particle_type, NodeType.SIZE)
            node_features = jnp.concatenate([node_features, particles], axis=-1)

        edge_features = [features[k] for k in ["rel_disp", "rel_dist"] if k in features]
        edge_features = jnp.concatenate(edge_features, axis=-1)

        feature_graph = jraph.GraphsTuple(
            nodes=IrrepsArray(self.node_features_irreps, node_features),
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
                self.edge_features_irreps, edge_features
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
        st_graph, dim = self._transform(*sample)
        st_graph = self.embedding(st_graph)
        for layer in self.layers:
            st_graph = layer(st_graph)
        nodes = self.decoder(st_graph)
        return self._postprocess(nodes, dim)
