"""
E(n) equivariant GNN  from `Garcia Satorras et al. <https://arxiv.org/abs/2102.09844>`_.
EGNN model, layers and feature transform.

Original implementation: https://github.com/vgsatorras/egnn

Standalone implementation + validation: https://github.com/gerkone/egnn-jax
"""

from typing import Callable, Dict, List, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jraph
from jax_sph.jax_md import space

from lagrangebench.utils import NodeType

from .base import BaseModel
from .utils import LinearXav, MLPXav


def _segment_sum(messages, indices, num_segments):
    return jraph.segment_sum(messages, indices, num_segments)


class _PosCorrectionMLP(eqx.Module):
    """Per-edge coord-correction MLP: blocks hidden Linear layers then a scalar head.

    The final layer's init variance is scaled by ``dt`` (UniformScaling(dt) in haiku).
    """

    hidden: List[LinearXav]
    head: LinearXav
    activation: Callable = eqx.field(static=True)
    use_tanh: bool = eqx.field(static=True)

    def __init__(
        self,
        in_size: int,
        hidden_size: int,
        blocks: int,
        activation: Callable,
        dt: float,
        use_tanh: bool,
        *,
        key,
    ):
        keys = jax.random.split(key, blocks + 1)
        sizes = [in_size] + [hidden_size] * blocks
        self.hidden = [
            LinearXav(sizes[i], sizes[i + 1], key=keys[i]) for i in range(blocks)
        ]
        self.head = LinearXav(
            hidden_size,
            1,
            with_bias=False,
            gain=dt,
            init="uniform_scaling",
            key=keys[-1],
        )
        self.activation = activation
        self.use_tanh = use_tanh

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.hidden:
            x = layer(x)
        x = self.activation(x)
        x = self.head(x)
        if self.use_tanh:
            x = jax.nn.tanh(x)
        return x


class _VelCorrectionMLP(eqx.Module):
    hidden: List[LinearXav]
    head: LinearXav
    activation: Callable = eqx.field(static=True)

    def __init__(
        self,
        in_size: int,
        hidden_size: int,
        blocks: int,
        activation: Callable,
        dt: float,
        *,
        key,
    ):
        keys = jax.random.split(key, blocks + 1)
        sizes = [in_size] + [hidden_size] * blocks
        self.hidden = [
            LinearXav(sizes[i], sizes[i + 1], key=keys[i]) for i in range(blocks)
        ]
        self.head = LinearXav(
            hidden_size,
            1,
            with_bias=False,
            gain=dt,
            init="uniform_scaling",
            key=keys[-1],
        )
        self.activation = activation

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.hidden:
            x = layer(x)
        x = self.activation(x)
        return self.head(x)


class _AttentionMLP(eqx.Module):
    linear: LinearXav

    def __init__(self, in_size: int, hidden_size: int, *, key):
        self.linear = LinearXav(in_size, hidden_size, key=key)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jax.nn.sigmoid(self.linear(x))


class EGNNLayer(eqx.Module):
    r"""E(n)-equivariant EGNN layer.

    Applies a message passing step where the positions are corrected with the velocities
    and a learnable correction term :math:`\psi_x(\mathbf{h}_i^{(t+1)})`:
    """

    edge_mlp: MLPXav
    node_mlp: MLPXav
    pos_mlp: _PosCorrectionMLP
    vel_mlp: _VelCorrectionMLP
    attention_mlp: Optional[_AttentionMLP]

    displacement_fn: Callable = eqx.field(static=True)
    shift_fn: Callable = eqx.field(static=True)
    residual: bool = eqx.field(static=True)
    normalize: bool = eqx.field(static=True)
    pos_aggregate_fn: Callable = eqx.field(static=True)
    msg_aggregate_fn: Callable = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        displacement_fn: Callable,
        shift_fn: Callable,
        edge_in_size: int,
        node_in_size: int,
        has_edge_attr: bool,
        has_node_attr: bool,
        blocks: int = 1,
        act_fn: Callable = jax.nn.silu,
        residual: bool = True,
        attention: bool = False,
        normalize: bool = False,
        tanh: bool = False,
        dt: float = 0.001,
        eps: float = 1e-8,
        pos_aggregate_fn: Callable = _segment_sum,
        msg_aggregate_fn: Callable = _segment_sum,
        *,
        key,
    ):
        """Initialize the layer.

        Args:
            hidden_size: hidden size
            output_size: output size
            displacement_fn: Displacement function for the acceleration computation.
            shift_fn: Shift function for updating positions
            edge_in_size: Size of ``edge_attr`` features fed into the edge MLP.
            node_in_size: Size of the scalar node features fed into the edge MLP.
            has_edge_attr: Whether an ``edge_attr`` tensor is concatenated to the
                edge input (``rel_dist`` in practice).
            has_node_attr: Whether a ``node_attr`` tensor is concatenated to the
                node input.
            blocks: number of blocks in the node and edge MLPs
            act_fn: activation function
            residual: whether to use residual connections
            attention: whether to use attention
            normalize: whether to normalize the coordinates
            tanh: whether to use tanh in the position update
            dt: position update step size
            eps: small number to avoid division by zero
            pos_aggregate_fn: position aggregation function
            msg_aggregate_fn: message aggregation function
            key: PRNG key for initialization.
        """
        self.displacement_fn = displacement_fn
        self.shift_fn = shift_fn
        self.residual = residual
        self.normalize = normalize
        self.pos_aggregate_fn = pos_aggregate_fn
        self.msg_aggregate_fn = msg_aggregate_fn
        self.eps = eps

        k_e, k_n, k_p, k_v, k_a = jax.random.split(key, 5)

        # edge mlp: in = 2 * nodes + radial + (edge_attr)
        edge_mlp_in = 2 * node_in_size + 1 + (edge_in_size if has_edge_attr else 0)
        self.edge_mlp = MLPXav(
            input_size=edge_mlp_in,
            output_sizes=[hidden_size] * blocks + [hidden_size],
            activation=act_fn,
            activate_final=True,
            key=k_e,
        )
        # node mlp: in = node + msg + (node_attr)
        node_mlp_in = node_in_size + hidden_size + (1 if has_node_attr else 0)
        self.node_mlp = MLPXav(
            input_size=node_mlp_in,
            output_sizes=[hidden_size] * blocks + [output_size],
            activation=act_fn,
            activate_final=False,
            key=k_n,
        )
        self.pos_mlp = _PosCorrectionMLP(
            in_size=hidden_size,
            hidden_size=hidden_size,
            blocks=blocks,
            activation=act_fn,
            dt=dt,
            use_tanh=tanh,
            key=k_p,
        )
        self.vel_mlp = _VelCorrectionMLP(
            in_size=output_size,
            hidden_size=hidden_size,
            blocks=blocks,
            activation=act_fn,
            dt=dt,
            key=k_v,
        )
        self.attention_mlp = (
            _AttentionMLP(hidden_size, hidden_size, key=k_a) if attention else None
        )

    def _coord2radial(
        self, senders: jnp.ndarray, receivers: jnp.ndarray, coord: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        coord_diff = self.displacement_fn(coord[senders], coord[receivers])
        radial = jnp.sum(coord_diff**2, 1)[:, jnp.newaxis]
        if self.normalize:
            norm = jnp.sqrt(radial)
            coord_diff = coord_diff / (norm + self.eps)
        return radial, coord_diff

    def __call__(
        self,
        nodes: jnp.ndarray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
        pos: jnp.ndarray,
        vel: jnp.ndarray,
        edge_attr: Optional[jnp.ndarray] = None,
        node_attr: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        radial, coord_diff = self._coord2radial(senders, receivers, pos)

        msg_inputs = [nodes[senders], nodes[receivers], radial]
        if edge_attr is not None:
            msg_inputs.append(edge_attr)
        msg = self.edge_mlp(jnp.concatenate(msg_inputs, axis=-1))
        if self.attention_mlp is not None:
            msg = msg * self.attention_mlp(msg)

        aggr = self.msg_aggregate_fn(msg, receivers, nodes.shape[0])
        node_inputs = [nodes, aggr]
        if node_attr is not None:
            node_inputs.append(node_attr)
        new_nodes = self.node_mlp(jnp.concatenate(node_inputs, axis=-1))
        if self.residual:
            new_nodes = nodes + new_nodes

        trans = coord_diff * self.pos_mlp(msg)
        pos_delta = self.pos_aggregate_fn(trans, senders, num_segments=pos.shape[0])
        pos = self.shift_fn(pos, pos_delta)
        pos = self.shift_fn(pos, self.vel_mlp(new_nodes) * vel)

        return new_nodes, pos


class EGNN(BaseModel):
    r"""
    E(n) Graph Neural Network by
    `Garcia Satorras et al. <https://arxiv.org/abs/2102.09844>`_.

    EGNN doesn't require expensive higher-order representations in intermediate layers;
    instead it relies on separate scalar and vector channels, which are treated
    differently by EGNN layers. In this setup, EGNN is similar to a learnable numerical
    integrator:

    .. math::
        \begin{align}
            \mathbf{m}_{ij}^{(t+1)} &= \phi_e \left(
                \mathbf{m}_{ij}^{(t)}, \mathbf{h}_i^{(t)},
                \mathbf{h}_j^{(t)}, ||\mathbf{x}_i^{(t)} - \mathbf{x}_j^{(t)}||^2
                \right) \\
            \mathbf{\hat{m}}_{ij}^{(t+1)} &=
            (\mathbf{x}_i^{(t)} - \mathbf{x}_j^{(t)}) \phi_x(\mathbf{m}_{ij}^{(t+1)})
        \end{align}

    And the node update with the integrator

    .. math::
        \begin{align}
            \mathbf{h}_i^{(t+1)} &= \psi_h \left(
                \mathbf{h}_i^{(t)}, \sum_{j \in \mathcal{N}(i)} \mathbf{m}_{ij}^{(t+1)}
                \right) \\
            \mathbf{x}_i^{(t+1)} &= \mathbf{x}_i^{(t)}
                + \mathbf{\hat{m}}_{ij}^{(t+1)} \psi_x(\mathbf{h}_i^{(t+1)})
        \end{align}

    where :math:`\mathbf{m}_{ij}` and :math:`\mathbf{\hat{m}}_{ij}` are the scalar and
    vector messages respectively, and :math:`\mathbf{x}_{i}` are the positions.

    This implementation differs from the original in two places:

    - because our datasets can have periodic boundary conditions, we use shift and
      displacement functions that take care of it when operations on positions are done.
    - we apply a simple integrator after the last layer to get the acceleration.
    """

    node_embed: LinearXav
    layers: List[EGNNLayer]

    output_size: int = eqx.field(static=True)
    num_mp_steps: int = eqx.field(static=True)
    n_vels: int = eqx.field(static=True)
    homogeneous_particles: bool = eqx.field(static=True)
    vel_mean: jnp.ndarray
    vel_std: jnp.ndarray
    displacement_fn: Callable = eqx.field(static=True)
    shift_fn: Callable = eqx.field(static=True)

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        dt: float,
        n_vels: int,
        displacement_fn: Callable,
        shift_fn: Callable,
        normalization_stats: Optional[Dict[str, Dict[str, jnp.ndarray]]] = None,
        act_fn: Callable = jax.nn.silu,
        num_mp_steps: int = 4,
        homogeneous_particles: bool = True,
        residual: bool = True,
        attention: bool = False,
        normalize: bool = False,
        tanh: bool = False,
        *,
        key,
    ):
        r"""
        Initialize the network.

        Args:
            hidden_size: Number of hidden features.
            output_size: Number of features for 'h' at the output.
            dt: Time step for position and velocity integration. Used to rescale the
                initialization of the correction MLP.
            n_vels: Number of velocities in the history.
            displacement_fn: Displacement function for the acceleration computation.
            shift_fn: Shift function for updating positions.
            normalization_stats: Normalization statistics for the input data.
            act_fn: Non-linearity.
            num_mp_steps: Number of layer for the EGNN
            homogeneous_particles: If all particles are of homogeneous type.
            residual: Whether to use residual connections.
            attention: Whether to use attention or not.
            normalize: Normalizes the coordinates messages such that:
                ``x^{l+1}_i = x^{l}_i + \sum(x_i - x_j)\phi_x(m_{ij})\|x_i - x_j\|``
                It may help in the stability or generalization. Not used in the paper.
            tanh: Sets a tanh activation function at the output of ``\phi_x(m_{ij})``.
                It bounds the output of ``\phi_x(m_{ij})`` which definitely improves in
                stability but it may decrease in accuracy. Not used in the paper.
            key: PRNG key for initialization.
        """
        self.output_size = output_size
        self.num_mp_steps = num_mp_steps
        self.n_vels = n_vels
        self.homogeneous_particles = homogeneous_particles
        self.displacement_fn = displacement_fn
        self.shift_fn = shift_fn

        if normalization_stats is None:
            normalization_stats = {
                "velocity": {"mean": jnp.array(0.0), "std": jnp.array(1.0)},
                "acceleration": {"mean": jnp.array(0.0), "std": jnp.array(1.0)},
            }
        self.vel_mean = jnp.asarray(normalization_stats["velocity"]["mean"])
        self.vel_std = jnp.asarray(normalization_stats["velocity"]["std"])

        # node scalar features: one magnitude per historical velocity
        node_scalar_size = n_vels
        if not homogeneous_particles:
            node_scalar_size += NodeType.SIZE

        keys = jax.random.split(key, num_mp_steps + 1)
        self.node_embed = LinearXav(node_scalar_size, hidden_size, key=keys[0])

        dt_per_step = dt / num_mp_steps
        self.layers = [
            EGNNLayer(
                hidden_size=hidden_size,
                output_size=hidden_size,
                displacement_fn=displacement_fn,
                shift_fn=shift_fn,
                edge_in_size=1,
                node_in_size=hidden_size,
                has_edge_attr=True,  # rel_dist
                has_node_attr=False,
                act_fn=act_fn,
                residual=residual,
                attention=attention,
                normalize=normalize,
                tanh=tanh,
                dt=dt_per_step,
                key=keys[i + 1],
            )
            for i in range(num_mp_steps)
        ]

    def _transform(
        self, features: Dict[str, jnp.ndarray], particle_type: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray], Optional[jnp.ndarray]]:
        n_nodes = features["vel_hist"].shape[0]
        vel = jnp.reshape(features["vel_hist"], (n_nodes, self.n_vels, -1))
        pos = features["abs_pos"][:, -1]
        edge_attr = features["rel_dist"]

        node_attr = None
        if "force" in features:
            node_attr = jnp.sqrt(
                jnp.sum(features["force"] ** 2, axis=-1, keepdims=True)
            )

        node_features = jnp.concatenate(
            [
                jnp.sqrt(jnp.sum(vel[:, i, :] ** 2, axis=-1, keepdims=True))
                for i in range(self.n_vels)
            ],
            axis=-1,
        )
        if not self.homogeneous_particles:
            particles = jax.nn.one_hot(particle_type, NodeType.SIZE)
            node_features = jnp.concatenate([node_features, particles], axis=-1)

        return (
            node_features,
            vel,
            pos,
            features["senders"],
            features["receivers"],
            edge_attr,
            node_attr,
        )

    def __call__(
        self, sample: Tuple[Dict[str, jnp.ndarray], jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        features, particle_type = sample
        (
            nodes,
            vel_hist,
            pos,
            senders,
            receivers,
            edge_attr,
            node_attr,
        ) = self._transform(features, particle_type)

        prev_vel_norm = vel_hist[:, -1, :]
        # EGNN works with unnormalized velocities
        prev_vel = prev_vel_norm * self.vel_std + self.vel_mean

        h = self.node_embed(nodes)
        next_pos = pos
        for layer in self.layers:
            h, next_pos = layer(
                nodes=h,
                senders=senders,
                receivers=receivers,
                pos=next_pos,
                vel=prev_vel,
                edge_attr=edge_attr,
                node_attr=node_attr,
            )

        next_vel = self.displacement_fn(next_pos, pos)
        acc = next_vel - prev_vel
        return {"pos": next_pos, "vel": next_vel, "acc": acc}
