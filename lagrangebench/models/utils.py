from typing import Callable, Dict, Iterable, NamedTuple, Optional

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp
import jraph

from lagrangebench.utils import NodeType


class LinearXav(hk.Linear):
    """Linear layer with Xavier init. Avoid distracting 'w_init' everywhere."""

    def __init__(
        self,
        output_size: int,
        with_bias: bool = True,
        w_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
        name: Optional[str] = None,
    ):
        if w_init is None:
            w_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")
        super().__init__(output_size, with_bias, w_init, b_init, name)


class MLPXav(hk.nets.MLP):
    """MLP layer with Xavier init. Avoid distracting 'w_init' everywhere."""

    def __init__(
        self,
        output_sizes: Iterable[int],
        with_bias: bool = True,
        w_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
        activation: Callable = jax.nn.silu,
        activate_final: bool = False,
        name: Optional[str] = None,
    ):
        if w_init is None:
            w_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")
        if not with_bias:
            b_init = None
        super().__init__(
            output_sizes,
            w_init,
            b_init,
            with_bias,
            activation,
            activate_final,
            name,
        )


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


def build_mlp(
    latent_size, output_size, num_hidden_layers, is_layer_norm=True, **kwds: Dict
):
    """MLP generation helper using Haiku."""
    assert num_hidden_layers >= 1
    network = hk.nets.MLP(
        [latent_size] * (num_hidden_layers - 1) + [output_size],
        **kwds,
        activate_final=False,
        name="MLP",
    )
    if is_layer_norm:
        l_norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        return hk.Sequential([network, l_norm])
    else:
        return network


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
