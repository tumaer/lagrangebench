"""
E(n) equivariant GNN  from `Garcia Satorras et al. <https://arxiv.org/abs/2102.09844>`_.
EGNN model, layers and feature transform.

Original implementation: https://github.com/vgsatorras/egnn

Standalone implementation + validation: https://github.com/gerkone/egnn-jax
"""

from typing import Any, Callable, Dict, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jraph
from jax.tree_util import Partial
from jax_sph.jax_md import space

from lagrangebench.utils import NodeType

from .base import BaseModel
from .utils import LinearXav, MLPXav


class EGNNLayer(hk.Module):
    r"""E(n)-equivariant EGNN layer.

    Applies a message passing step where the positions are corrected with the velocities
    and a learnable correction term :math:`\psi_x(\mathbf{h}_i^{(t+1)})`:
    """

    def __init__(
        self,
        layer_num: int,
        hidden_size: int,
        output_size: int,
        displacement_fn: space.DisplacementFn,
        shift_fn: space.ShiftFn,
        blocks: int = 1,
        act_fn: Callable = jax.nn.silu,
        pos_aggregate_fn: Optional[Callable] = jraph.segment_sum,
        msg_aggregate_fn: Optional[Callable] = jraph.segment_sum,
        residual: bool = True,
        attention: bool = False,
        normalize: bool = False,
        tanh: bool = False,
        dt: float = 0.001,
        eps: float = 1e-8,
    ):
        """Initialize the layer.

        Args:
            layer_num: layer number
            hidden_size: hidden size
            output_size: output size
            displacement_fn: Displacement function for the acceleration computation.
            shift_fn: Shift function for updating positions
            blocks: number of blocks in the node and edge MLPs
            act_fn: activation function
            pos_aggregate_fn: position aggregation function
            msg_aggregate_fn: message aggregation function
            residual: whether to use residual connections
            attention: whether to use attention
            normalize: whether to normalize the coordinates
            tanh: whether to use tanh in the position update
            dt: position update step size
            eps: small number to avoid division by zero
        """
        super().__init__(f"layer_{layer_num}")

        self._displacement_fn = displacement_fn
        self._shift_fn = shift_fn
        self.pos_aggregate_fn = pos_aggregate_fn
        self.msg_aggregate_fn = msg_aggregate_fn
        self._residual = residual
        self._normalize = normalize
        self._eps = eps

        # message network
        self._edge_mlp = MLPXav(
            [hidden_size] * blocks + [hidden_size],
            activation=act_fn,
            activate_final=True,
        )

        # update network
        self._node_mlp = MLPXav(
            [hidden_size] * blocks + [output_size],
            activation=act_fn,
            activate_final=False,
        )

        # position update network
        net = [LinearXav(hidden_size)] * blocks
        # NOTE: from https://github.com/vgsatorras/egnn/blob/main/models/gcl.py#L254
        net += [
            act_fn,
            LinearXav(1, with_bias=False, w_init=hk.initializers.UniformScaling(dt)),
        ]
        if tanh:
            net.append(jax.nn.tanh)
        self._pos_correction_mlp = hk.Sequential(net)

        # velocity integrator network
        net = [LinearXav(hidden_size)] * blocks
        net += [
            act_fn,
            LinearXav(1, with_bias=False, w_init=hk.initializers.UniformScaling(dt)),
        ]
        self._vel_correction_mlp = hk.Sequential(net)

        # attention
        self._attention_mlp = None
        if attention:
            self._attention_mlp = hk.Sequential(
                [LinearXav(hidden_size), jax.nn.sigmoid]
            )

    def _pos_update(
        self,
        pos: jnp.ndarray,
        graph: jraph.GraphsTuple,
        coord_diff: jnp.ndarray,
    ) -> jnp.ndarray:
        trans = coord_diff * self._pos_correction_mlp(graph.edges)
        return self.pos_aggregate_fn(trans, graph.senders, num_segments=pos.shape[0])

    def _message(
        self,
        radial: jnp.ndarray,
        edge_attribute: jnp.ndarray,
        edge_features: Any,
        incoming: jnp.ndarray,
        outgoing: jnp.ndarray,
        globals_: Any,
    ) -> jnp.ndarray:
        _ = edge_features
        _ = globals_
        msg = jnp.concatenate([incoming, outgoing, radial], axis=-1)
        if edge_attribute is not None:
            msg = jnp.concatenate([msg, edge_attribute], axis=-1)
        msg = self._edge_mlp(msg)
        if self._attention_mlp:
            att = self._attention_mlp(msg)
            msg = msg * att
        return msg

    def _update(
        self,
        node_attribute: jnp.ndarray,
        nodes: jnp.ndarray,
        senders: Any,
        msg: jnp.ndarray,
        globals_: Any,
    ) -> jnp.ndarray:
        _ = senders
        _ = globals_
        x = jnp.concatenate([nodes, msg], axis=-1)
        if node_attribute is not None:
            x = jnp.concatenate([x, node_attribute], axis=-1)
        x = self._node_mlp(x)
        if self._residual:
            x = nodes + x
        return x

    def _coord2radial(
        self, graph: jraph.GraphsTuple, coord: jnp.array
    ) -> Tuple[jnp.array, jnp.array]:
        coord_diff = self._displacement_fn(coord[graph.senders], coord[graph.receivers])
        radial = jnp.sum(coord_diff**2, 1)[:, jnp.newaxis]
        if self._normalize:
            norm = jnp.sqrt(radial)
            coord_diff = coord_diff / (norm + self._eps)
        return radial, coord_diff

    def __call__(
        self,
        graph: jraph.GraphsTuple,
        pos: jnp.ndarray,
        vel: jnp.ndarray,
        edge_attribute: Optional[jnp.ndarray] = None,
        node_attribute: Optional[jnp.ndarray] = None,
    ) -> Tuple[jraph.GraphsTuple, jnp.ndarray]:
        """
        Apply EGNN layer.

        Args:
            graph: Graph from previous step
            pos: Node position, updated separately
            vel: Node velocity
            edge_attribute: Edge attribute (optional)
            node_attribute: Node attribute (optional)
        Returns:
            Updated graph, node position
        """
        radial, coord_diff = self._coord2radial(graph, pos)
        graph = jraph.GraphNetwork(
            update_edge_fn=Partial(self._message, radial, edge_attribute),
            update_node_fn=Partial(self._update, node_attribute),
            aggregate_edges_for_nodes_fn=self.msg_aggregate_fn,
        )(graph)
        # update position
        pos = self._shift_fn(pos, self._pos_update(pos, graph, coord_diff))
        # integrate velocity
        pos = self._shift_fn(pos, self._vel_correction_mlp(graph.nodes) * vel)
        return graph, pos


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

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        dt: float,
        n_vels: int,
        displacement_fn: space.DisplacementFn,
        shift_fn: space.ShiftFn,
        normalization_stats: Optional[Dict[str, jnp.ndarray]] = None,
        act_fn: Callable = jax.nn.silu,
        num_mp_steps: int = 4,
        homogeneous_particles: bool = True,
        residual: bool = True,
        attention: bool = False,
        normalize: bool = False,
        tanh: bool = False,
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
        """
        super().__init__()
        # network
        self._hidden_size = hidden_size
        self._output_size = output_size
        self._act_fn = act_fn
        self._num_mp_steps = num_mp_steps
        self._residual = residual
        self._attention = attention
        self._normalize = normalize
        self._tanh = tanh

        # integrator
        self._dt = dt / self._num_mp_steps
        self._displacement_fn = displacement_fn
        self._shift_fn = shift_fn
        if normalization_stats is None:
            normalization_stats = {
                "velocity": {"mean": 0.0, "std": 1.0},
                "acceleration": {"mean": 0.0, "std": 1.0},
            }
        self._vel_stats = normalization_stats["velocity"]
        self._acc_stats = normalization_stats["acceleration"]

        # transform
        self._n_vels = n_vels
        self._homogeneous_particles = homogeneous_particles

    def _transform(
        self, features: Dict[str, jnp.ndarray], particle_type: jnp.ndarray
    ) -> Tuple[jraph.GraphsTuple, Dict[str, jnp.ndarray]]:
        props = {}
        n_nodes = features["vel_hist"].shape[0]

        props["vel"] = jnp.reshape(features["vel_hist"], (n_nodes, self._n_vels, -1))

        # most recent position
        props["pos"] = features["abs_pos"][:, -1]
        # relative distances between particles
        props["edge_attr"] = features["rel_dist"]
        # force magnitude as node attributes
        props["node_attr"] = None
        if "force" in features:
            props["node_attr"] = jnp.sqrt(
                jnp.sum(features["force"] ** 2, axis=-1, keepdims=True)
            )

        # velocity magnitudes as node features
        node_features = jnp.concatenate(
            [
                jnp.sqrt(jnp.sum(props["vel"][:, i, :] ** 2, axis=-1, keepdims=True))
                for i in range(self._n_vels)
            ],
            axis=-1,
        )
        if not self._homogeneous_particles:
            particles = jax.nn.one_hot(particle_type, NodeType.SIZE)
            node_features = jnp.concatenate([node_features, particles], axis=-1)

        graph = jraph.GraphsTuple(
            nodes=node_features,
            edges=None,
            senders=features["senders"],
            receivers=features["receivers"],
            n_node=jnp.array([n_nodes]),
            n_edge=jnp.array([len(features["senders"])]),
            globals=None,
        )

        return graph, props

    def _postprocess(
        self, next_pos: jnp.ndarray, props: Dict[str, jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        prev_vel = props["vel"][:, -1, :]
        prev_pos = props["pos"]
        # first order finite difference
        next_vel = self._displacement_fn(next_pos, prev_pos)
        acc = next_vel - prev_vel
        return {"pos": next_pos, "vel": next_vel, "acc": acc}

    def __call__(
        self, sample: Tuple[Dict[str, jnp.ndarray], jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        graph, props = self._transform(*sample)
        # input node embedding
        h = LinearXav(self._hidden_size, name="scalar_emb")(graph.nodes)
        graph = graph._replace(nodes=h)
        prev_vel = props["vel"][:, -1, :]
        # egnn works with unnormalized velocities
        prev_vel = prev_vel * self._vel_stats["std"] + self._vel_stats["mean"]
        # message passing
        next_pos = props["pos"].copy()
        for n in range(self._num_mp_steps):
            graph, next_pos = EGNNLayer(
                layer_num=n,
                hidden_size=self._hidden_size,
                output_size=self._hidden_size,
                displacement_fn=self._displacement_fn,
                shift_fn=self._shift_fn,
                act_fn=self._act_fn,
                residual=self._residual,
                attention=self._attention,
                normalize=self._normalize,
                dt=self._dt,
                tanh=self._tanh,
            )(graph, next_pos, prev_vel, props["edge_attr"], props["node_attr"])

        # position finite differencing to get acceleration
        out = self._postprocess(next_pos, props)
        return out
