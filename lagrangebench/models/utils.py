"""Model utilities and building blocks (Equinox-based)."""

from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import e3nn_jax as e3nn
import equinox as eqx
import jax
import jax.numpy as jnp
import jraph
from jax import Array

from lagrangebench.utils import NodeType


def _xavier_uniform(key, shape: Tuple[int, ...], gain: float = 1.0) -> Array:
    """Xavier/Glorot uniform over a (in, out) weight shape."""
    fan_in, fan_out = shape[-2], shape[-1]
    limit = gain * jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)


def _uniform_scaling(key, shape: Tuple[int, ...], scale: float = 1.0) -> Array:
    """Uniform-scaling init: ``U(-limit, +limit)`` with
    ``limit = scale * sqrt(3 / fan_in)``. Matches haiku's ``UniformScaling``.
    """
    fan_in = shape[-2]
    limit = scale * jnp.sqrt(3.0 / fan_in)
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)


class LinearXav(eqx.Module):
    """Linear layer with Xavier/Glorot-uniform init, applied on the last axis."""

    weight: Array
    bias: Optional[Array]

    def __init__(
        self,
        in_size: int,
        out_size: int,
        with_bias: bool = True,
        gain: float = 1.0,
        init: str = "xavier_avg",
        *,
        key,
    ):
        """Initialize.

        Args:
            in_size: Input feature size (last-axis length).
            out_size: Output feature size.
            with_bias: Whether to include a bias term.
            gain: Multiplicative factor on the initialization limit.
            init: ``"xavier_avg"`` uses ``gain * sqrt(6/(fan_in+fan_out))``;
                ``"uniform_scaling"`` uses ``gain * sqrt(3/fan_in)`` (equivalent
                to haiku's ``hk.initializers.UniformScaling(gain)``).
            key: PRNG key for initialization.
        """
        if init == "xavier_avg":
            self.weight = _xavier_uniform(key, (in_size, out_size), gain=gain)
        elif init == "uniform_scaling":
            self.weight = _uniform_scaling(key, (in_size, out_size), scale=gain)
        else:
            raise ValueError(f"Unknown init: {init}")
        self.bias = jnp.zeros(out_size) if with_bias else None

    def __call__(self, x: Array) -> Array:
        y = x @ self.weight
        if self.bias is not None:
            y = y + self.bias
        return y


class LayerNorm(eqx.Module):
    """LayerNorm on the last axis."""

    scale: Array
    offset: Array
    eps: float = eqx.field(static=True)

    def __init__(self, size: int, eps: float = 1e-5):
        self.scale = jnp.ones(size)
        self.offset = jnp.zeros(size)
        self.eps = eps

    def __call__(self, x: Array) -> Array:
        mean = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)
        return (x - mean) * jax.lax.rsqrt(var + self.eps) * self.scale + self.offset


class MLPXav(eqx.Module):
    """MLP with Xavier-init dense layers. Analogue of haiku's hk.nets.MLP."""

    layers: List[LinearXav]
    activation: Callable = eqx.field(static=True)
    activate_final: bool = eqx.field(static=True)

    def __init__(
        self,
        input_size: int,
        output_sizes: List[int],
        with_bias: bool = True,
        activation: Callable = jax.nn.silu,
        activate_final: bool = False,
        *,
        key,
    ):
        keys = jax.random.split(key, len(output_sizes))
        sizes = [input_size] + list(output_sizes)
        self.layers = [
            LinearXav(sizes[i], sizes[i + 1], with_bias=with_bias, key=keys[i])
            for i in range(len(output_sizes))
        ]
        self.activation = activation
        self.activate_final = activate_final

    def __call__(self, x: Array) -> Array:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1 or self.activate_final:
                x = self.activation(x)
        return x


class MLPBlock(eqx.Module):
    """An MLP optionally followed by a LayerNorm. Replaces haiku's `build_mlp`."""

    mlp: MLPXav
    norm: Optional[LayerNorm]

    def __init__(
        self,
        input_size: int,
        latent_size: int,
        output_size: int,
        num_hidden_layers: int,
        is_layer_norm: bool = True,
        activation: Callable = jax.nn.silu,
        *,
        key,
    ):
        assert num_hidden_layers >= 1
        output_sizes = [latent_size] * (num_hidden_layers - 1) + [output_size]
        self.mlp = MLPXav(
            input_size=input_size,
            output_sizes=output_sizes,
            activation=activation,
            activate_final=False,
            key=key,
        )
        self.norm = LayerNorm(output_size) if is_layer_norm else None

    def __call__(self, x: Array) -> Array:
        x = self.mlp(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


def build_mlp(
    input_size: int,
    latent_size: int,
    output_size: int,
    num_hidden_layers: int,
    is_layer_norm: bool = True,
    activation: Callable = jax.nn.silu,
    *,
    key,
) -> MLPBlock:
    """MLP factory: stacks ``num_hidden_layers`` dense layers + optional LayerNorm."""
    return MLPBlock(
        input_size=input_size,
        latent_size=latent_size,
        output_size=output_size,
        num_hidden_layers=num_hidden_layers,
        is_layer_norm=is_layer_norm,
        activation=activation,
        key=key,
    )


class Embedding(eqx.Module):
    """Categorical embedding table (jax-friendly replacement for hk.Embed)."""

    table: Array

    def __init__(self, vocab_size: int, embed_size: int, *, key):
        self.table = jax.random.truncated_normal(
            key, lower=-2.0, upper=2.0, shape=(vocab_size, embed_size)
        ) * 0.02

    def __call__(self, ids: Array) -> Array:
        return self.table[ids]


class SteerableGraphsTuple(NamedTuple):
    r"""
    Pack (steerable) node and edge attributes with jraph.GraphsTuple.

    Attributes:
        graph: jraph.GraphsTuple, graph structure
        node_attributes: (N, irreps.dim), node attributes :math:`\mathbf{\hat{a}}_i`
        edge_attributes: (E, irreps.dim), edge attributes :math:`\mathbf{\hat{a}}_{ij}`
        additional_message_features: (E, edge_dim), optional message features
    """

    graph: jraph.GraphsTuple
    node_attributes: Optional[e3nn.IrrepsArray] = None
    edge_attributes: Optional[e3nn.IrrepsArray] = None
    # NOTE: additional_message_features is in a separate field otherwise it would get
    #  updated by jraph.GraphNetwork. Actual graph edges are used only for the messages.
    additional_message_features: Optional[e3nn.IrrepsArray] = None


def node_irreps(
    metadata: Dict,
    input_seq_length: int,
    has_external_force: bool,
    has_magnitudes: bool,
    has_homogeneous_particles: bool,
) -> str:
    """Compute input node irreps based on which features are available."""
    irreps = []
    irreps.append(f"{input_seq_length - 1}x1o")
    if not any(metadata["periodic_boundary_conditions"]):
        irreps.append("2x1o")
    if has_external_force:
        irreps.append("1x1o")
    if has_magnitudes:
        irreps.append(f"{input_seq_length - 1}x0e")
    if not has_homogeneous_particles:
        irreps.append(f"{NodeType.SIZE}x0e")
    return e3nn.Irreps("+".join(irreps))


def features_2d_to_3d(features):
    """Add zeros in the z component of 2D features."""
    n_nodes = features["vel_hist"].shape[0]
    n_edges = features["rel_disp"].shape[0]
    n_vels = features["vel_hist"].shape[1]
    features["vel_hist"] = jnp.concatenate(
        [features["vel_hist"], jnp.zeros((n_nodes, n_vels, 1))], -1
    )
    features["rel_disp"] = jnp.concatenate(
        [features["rel_disp"], jnp.zeros((n_edges, 1))], -1
    )
    if "bound" in features:
        features["bound"] = jnp.concatenate(
            [features["bound"], jnp.zeros((n_nodes, 1))], -1
        )
    if "force" in features:
        features["force"] = jnp.concatenate(
            [features["force"], jnp.zeros((n_nodes, 1))], -1
        )
    return features


def node_feature_size(
    metadata: Dict,
    input_seq_length: int,
    has_external_force: bool,
    has_magnitudes: bool,
    has_homogeneous_particles: bool,
) -> int:
    """Scalar-feature size for GNS (sum of vel_hist, vel_mag, bound, force, embed)."""
    dim = metadata["dim"]
    size = (input_seq_length - 1) * dim  # vel_hist
    if has_magnitudes:
        size += input_seq_length - 1  # vel_mag
    if not any(metadata["periodic_boundary_conditions"]):
        size += 2 * dim  # bound (distance to lower + upper)
    if has_external_force:
        size += dim  # force
    return size


def edge_feature_size(metadata: Dict) -> int:
    """Edge feature size (rel_disp + rel_dist)."""
    return metadata["dim"] + 1
