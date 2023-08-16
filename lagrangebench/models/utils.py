from typing import Dict, NamedTuple, Optional

import e3nn_jax as e3nn
import haiku as hk
import jax.numpy as jnp
import jraph

from lagrangebench.utils import NodeType


class SteerableGraphsTuple(NamedTuple):
    """Pack (steerable) node and edge attributes with jraph.GraphsTuple."""

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


def build_mlp(latent_size, output_size, num_layers, is_layer_norm=True, **kwds: Dict):
    """MLP generation helper using Haiku"""
    assert num_layers >= 1
    network = hk.nets.MLP(
        [latent_size] * (num_layers - 1) + [output_size],
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
