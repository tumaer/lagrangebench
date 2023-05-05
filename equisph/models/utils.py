from typing import Dict, NamedTuple, Optional

import e3nn_jax as e3nn
import haiku as hk
import jraph

from equisph.simulate import NodeType


class SteerableGraphsTuple(NamedTuple):
    """Pack (steerable) node and edge attributes with jraph.GraphsTuple."""

    graph: jraph.GraphsTuple
    node_attributes: Optional[e3nn.IrrepsArray] = None
    edge_attributes: Optional[e3nn.IrrepsArray] = None
    # NOTE: additional_message_features is in a separate field otherwise it would get
    #  updated by jraph.GraphNetwork. Actual graph edges are used only for the messages.
    additional_message_features: Optional[e3nn.IrrepsArray] = None


def node_irreps(args) -> str:
    irreps = []
    irreps.append(f"{args.config.input_seq_length - 1}x1o")
    if not any(args.metadata["periodic_boundary_conditions"]):
        irreps.append("2x1o")

    if args.info.has_external_force:
        irreps.append("1x1o")

    if args.config.magnitudes:
        irreps.append(f"{args.config.input_seq_length - 1}x0e")

    if not args.info.homogeneous_particles:
        irreps.append(f"{NodeType.SIZE}x0e")

    return "+".join(irreps)


def build_mlp(latent_size, output_size, num_layers, is_layer_norm=True, **kwds: Dict):
    """MLP generation helper using Haiku"""
    network = hk.nets.MLP(
        [latent_size] * num_layers + [output_size],
        **kwds,
        activate_final=False,
        name="MLP",
    )
    if is_layer_norm:
        l_norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        return hk.Sequential([network, l_norm])
    else:
        return network
