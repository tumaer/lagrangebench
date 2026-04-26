"""
Modified PaiNN implementation for general vectorial inputs and outputs
`Schütt et al. <https://proceedings.mlr.press/v139/schutt21a.html>`_.
PaiNN model, layers and feature transform.

Original implementation: https://github.com/atomistic-machine-learning/schnetpack

Standalone implementation + validation: https://github.com/gerkone/painn-jax
"""

from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jraph

from lagrangebench.utils import NodeType

from .base import BaseModel
from .utils import LinearXav


class NodeFeatures(NamedTuple):
    """Simple container for PaiNN scalar and vectorial node features."""

    s: jnp.ndarray = None
    v: jnp.ndarray = None


ReadoutFn = Callable[[jraph.GraphsTuple], Tuple[jnp.ndarray, jnp.ndarray]]
ReadoutBuilderFn = Callable[..., ReadoutFn]


class GatedEquivariantBlock(eqx.Module):
    """Gated equivariant block (restricted to vectorial features).

    .. image:: https://i.imgur.com/EMlg2Qi.png
    """

    vector_mix_net: LinearXav
    gate_lin1: LinearXav
    gate_lin2: LinearXav
    activation: Callable = eqx.field(static=True)
    scalar_activation: Optional[Callable] = eqx.field(static=True)
    scalar_out_channels: int = eqx.field(static=True)
    vector_out_channels: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def __init__(
        self,
        s_in_size: int,
        v_in_channels: int,
        hidden_size: int,
        scalar_out_channels: int,
        vector_out_channels: int,
        activation: Callable = jax.nn.silu,
        scalar_activation: Optional[Callable] = None,
        eps: float = 1e-8,
        *,
        key,
    ):
        """Initialize the layer.

        Args:
            s_in_size: Input scalar feature size.
            v_in_channels: Input vector feature channels.
            hidden_size: Number of hidden channels.
            scalar_out_channels: Number of scalar output channels.
            vector_out_channels: Number of vector output channels.
            activation: Gate activation function.
            scalar_activation: Activation function for the scalar output.
            eps: Constant added in norm to prevent derivation instabilities.
            key: PRNG key for initialization.
        """
        assert scalar_out_channels > 0 and vector_out_channels > 0
        self.scalar_out_channels = scalar_out_channels
        self.vector_out_channels = vector_out_channels
        self.eps = eps

        k_v, k_g1, k_g2 = jax.random.split(key, 3)
        self.vector_mix_net = LinearXav(
            v_in_channels, 2 * vector_out_channels, with_bias=False, key=k_v
        )
        self.gate_lin1 = LinearXav(
            s_in_size + vector_out_channels, hidden_size, key=k_g1
        )
        self.gate_lin2 = LinearXav(
            hidden_size, scalar_out_channels + vector_out_channels, key=k_g2
        )
        self.activation = activation
        self.scalar_activation = scalar_activation

    def __call__(
        self, s: jnp.ndarray, v: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        v_l, v_r = jnp.split(self.vector_mix_net(v), 2, axis=-1)

        v_r_norm = jnp.sqrt(jnp.sum(v_r**2, axis=-2) + self.eps)
        gating_scalars = jnp.concatenate([s, v_r_norm], axis=-1)
        gate_out = self.gate_lin2(self.activation(self.gate_lin1(gating_scalars)))
        s_new, _, v_gate = jnp.split(
            gate_out,
            [self.scalar_out_channels, self.vector_out_channels],
            axis=-1,
        )
        # scale the vectors by the gating scalars
        v_new = v_l * v_gate[:, jnp.newaxis]

        if self.scalar_activation is not None:
            s_new = self.scalar_activation(s_new)
        return s_new, v_new


class GaussianRBF(eqx.Module):
    r"""Gaussian radial basis functions.

    Args:
        n_rbf: total number of Gaussian functions, :math:`N_g`.
        cutoff: center of last Gaussian function, :math:`\mu_{N_g}`
        start: center of first Gaussian function, :math:`\mu_0`.
        trainable: If True, widths and offset of Gaussian functions learnable.
    """

    widths: jnp.ndarray
    offsets: jnp.ndarray
    trainable: bool = eqx.field(static=True)

    def __init__(
        self,
        n_rbf: int,
        cutoff: float,
        start: float = 0.0,
        centered: bool = False,
        trainable: bool = False,
    ):
        if centered:
            widths = jnp.linspace(start, cutoff, n_rbf)
            offsets = jnp.zeros_like(widths)
        else:
            offsets = jnp.linspace(start, cutoff, n_rbf)
            widths = jnp.abs(cutoff - start) / n_rbf * jnp.ones_like(offsets)
        self.widths = widths
        self.offsets = offsets
        self.trainable = trainable

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        widths = self.widths if self.trainable else jax.lax.stop_gradient(self.widths)
        offsets = (
            self.offsets if self.trainable else jax.lax.stop_gradient(self.offsets)
        )
        coeff = -0.5 / jnp.power(widths, 2)
        diff = x[..., jnp.newaxis] - offsets
        return jnp.exp(coeff * jnp.power(diff, 2))


def gaussian_rbf(
    n_rbf: int,
    cutoff: float,
    start: float = 0.0,
    centered: bool = False,
    trainable: bool = False,
) -> GaussianRBF:
    r"""Gaussian radial basis functions.

    Args:
        n_rbf: total number of Gaussian functions, :math:`N_g`.
        cutoff: center of last Gaussian function, :math:`\mu_{N_g}`
        start: center of first Gaussian function, :math:`\mu_0`.
        trainable: If True, widths and offset of Gaussian functions learnable.
    """
    return GaussianRBF(n_rbf, cutoff, start=start, centered=centered, trainable=trainable)


class CosineCutoff(eqx.Module):
    r"""Behler-style cosine cutoff.

    .. math::
        f(r) = \begin{cases}
        0.5 \times \left[1 + \cos\left(\frac{\pi r}{r_\text{cutoff}}\right)\right]
            & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}

    Args:
        cutoff (float): cutoff radius.
    """

    cutoff: float = eqx.field(static=True)

    def __init__(self, cutoff: float):
        self.cutoff = cutoff

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        cuts = 0.5 * (jnp.cos(x * jnp.pi / self.cutoff) + 1.0)
        mask = jnp.array(x < self.cutoff, dtype=jnp.float32)
        return cuts * mask


def cosine_cutoff(cutoff: float) -> CosineCutoff:
    r"""Behler-style cosine cutoff.

    Args:
        cutoff (float): cutoff radius.
    """
    return CosineCutoff(cutoff)


class _PaiNNReadout(eqx.Module):
    """PaiNN readout block."""

    blocks: List[GatedEquivariantBlock]

    def __init__(
        self,
        hidden_size: int,
        in_channels: int,
        out_channels: int,
        activation: Callable,
        blocks: int,
        eps: float,
        *,
        key,
    ):
        keys = jax.random.split(key, blocks)
        modules = []
        s_in = in_channels
        v_in = in_channels
        ith_hidden_size = hidden_size
        for i in range(blocks - 1):
            ith_hidden_size = hidden_size // 2 ** (i + 1)
            modules.append(
                GatedEquivariantBlock(
                    s_in_size=s_in,
                    v_in_channels=v_in,
                    hidden_size=ith_hidden_size * 2,
                    scalar_out_channels=ith_hidden_size,
                    vector_out_channels=ith_hidden_size,
                    activation=activation,
                    eps=eps,
                    key=keys[i],
                )
            )
            s_in = ith_hidden_size
            v_in = ith_hidden_size
        modules.append(
            GatedEquivariantBlock(
                s_in_size=s_in,
                v_in_channels=v_in,
                hidden_size=ith_hidden_size,
                scalar_out_channels=out_channels,
                vector_out_channels=out_channels,
                activation=activation,
                eps=eps,
                key=keys[-1],
            )
        )
        self.blocks = modules

    def __call__(self, graph: jraph.GraphsTuple) -> Tuple[jnp.ndarray, jnp.ndarray]:
        s, v = graph.nodes
        s = jnp.squeeze(s)
        for block in self.blocks:
            s, v = block(s, v)
        return jnp.squeeze(s), jnp.squeeze(v)


def PaiNNReadout(
    hidden_size: int,
    in_channels: int,
    out_channels: int = 1,
    activation: Callable = jax.nn.silu,
    blocks: int = 2,
    eps: float = 1e-8,
    *,
    key,
) -> _PaiNNReadout:
    """
    PaiNN readout block.

    Args:
        hidden_size: Number of hidden channels.
        in_channels: Input scalar/vector channels from the last interaction block.
        out_channels: Number of scalar/vector output channels.
        activation: Activation function.
        blocks: Number of readout blocks.
        eps: Small constant for norm stability.
        key: PRNG key for initialization.

    Returns:
        Configured readout module.
    """
    return _PaiNNReadout(
        hidden_size=hidden_size,
        in_channels=in_channels,
        out_channels=out_channels,
        activation=activation,
        blocks=blocks,
        eps=eps,
        key=key,
    )


class PaiNNLayer(eqx.Module):
    """PaiNN interaction block."""

    interaction_block_lin: List[LinearXav]
    mixing_block_lin: List[LinearXav]
    vector_mixing_block: LinearXav
    activation: Callable = eqx.field(static=True)
    aggregate_fn: Callable = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def __init__(
        self,
        hidden_size: int,
        activation: Callable = jax.nn.silu,
        blocks: int = 2,
        aggregate_fn: Callable = jraph.segment_sum,
        eps: float = 1e-8,
        *,
        key,
    ):
        """
        Initialize the PaiNN layer, made up of an interaction block and a mixing block.

        Args:
            hidden_size: Number of node features.
            activation: Activation function.
            blocks: Number of layers in the context networks.
            aggregate_fn: Function to aggregate the neighbors.
            eps: Constant added in norm to prevent derivation instabilities.
            key: PRNG key for initialization.
        """
        self.hidden_size = hidden_size
        self.eps = eps
        self.aggregate_fn = aggregate_fn
        self.activation = activation

        k_i, k_m, k_v = jax.random.split(key, 3)
        k_i_layers = jax.random.split(k_i, blocks)
        k_m_layers = jax.random.split(k_m, blocks)

        # inter-particle context net (interaction_block)
        ib_layers = []
        for i in range(blocks - 1):
            ib_layers.append(LinearXav(hidden_size, hidden_size, key=k_i_layers[i]))
        ib_layers.append(LinearXav(hidden_size, 3 * hidden_size, key=k_i_layers[-1]))
        self.interaction_block_lin = ib_layers

        # intra-particle context net (mixing_block)
        mb_layers = []
        # mixing input: [s (hidden), v_norm (hidden)] concat = 2 * hidden
        mb_layers.append(LinearXav(2 * hidden_size, hidden_size, key=k_m_layers[0]))
        for i in range(1, blocks - 1):
            mb_layers.append(LinearXav(hidden_size, hidden_size, key=k_m_layers[i]))
        mb_layers.append(LinearXav(hidden_size, 3 * hidden_size, key=k_m_layers[-1]))
        self.mixing_block_lin = mb_layers

        # vector channel mix
        self.vector_mixing_block = LinearXav(
            hidden_size, 2 * hidden_size, with_bias=False, key=k_v
        )

    def _interaction(self, s: jnp.ndarray) -> jnp.ndarray:
        x = s
        for i, layer in enumerate(self.interaction_block_lin):
            x = layer(x)
            if i < len(self.interaction_block_lin) - 1:
                x = self.activation(x)
        return x

    def _mixing(self, ts: jnp.ndarray) -> jnp.ndarray:
        x = ts
        for i, layer in enumerate(self.mixing_block_lin):
            x = layer(x)
            if i < len(self.mixing_block_lin) - 1:
                x = self.activation(x)
        return x

    def _message(
        self,
        s: jnp.ndarray,
        v: jnp.ndarray,
        dir_ij: jnp.ndarray,
        Wij: jnp.ndarray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Message/interaction. Inter-particle.

        Args:
            s (jnp.ndarray): Input scalar features.
            v (jnp.ndarray): Input vector features.
            dir_ij (jnp.ndarray): Direction of the edge.
            Wij (jnp.ndarray): Filter.
            senders (jnp.ndarray): Index of the sender node.
            receivers (jnp.ndarray): Index of the receiver node.

        Returns:
            Aggregated messages after interaction.
        """
        x = self._interaction(s)
        xj = x[receivers]
        vj = v[receivers]

        ds, dv1, dv2 = jnp.split(Wij * xj, 3, axis=-1)
        n_nodes = s.shape[0]
        dv = dv1 * dir_ij[..., jnp.newaxis] + dv2 * vj
        ds = self.aggregate_fn(ds, senders, n_nodes)
        dv = self.aggregate_fn(dv, senders, n_nodes)

        s = s + jnp.clip(ds, -1e2, 1e2)
        v = v + jnp.clip(dv, -1e2, 1e2)
        return s, v

    def _update(
        self, s: jnp.ndarray, v: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update/mixing. Intra-particle.

        Args:
            s (jnp.ndarray): Input scalar features.
            v (jnp.ndarray): Input vector features.

        Returns:
            Node features after update.
        """
        v_l, v_r = jnp.split(self.vector_mixing_block(v), 2, axis=-1)
        v_norm = jnp.sqrt(jnp.sum(v_r**2, axis=-2, keepdims=True) + self.eps)

        ts = jnp.concatenate([s, v_norm], axis=-1)
        ds, dv, dsv = jnp.split(self._mixing(ts), 3, axis=-1)
        dv = v_l * dv
        dsv = dsv * jnp.sum(v_r * v_l, axis=1, keepdims=True)

        s = s + jnp.clip(ds + dsv, -1e2, 1e2)
        v = v + jnp.clip(dv, -1e2, 1e2)
        return s, v

    def __call__(
        self,
        graph: jraph.GraphsTuple,
        Wij: jnp.ndarray,
    ):
        """Compute interaction output.

        Args:
            graph (jraph.GraphsTuple): Input graph.
            Wij (jnp.ndarray): Filter.

        Returns:
            atom features after interaction
        """
        s, v = graph.nodes
        s, v = self._message(s, v, graph.edges, Wij, graph.senders, graph.receivers)
        s, v = self._update(s, v)
        return graph._replace(nodes=NodeFeatures(s=s, v=v))


class PaiNN(BaseModel):
    r"""Polarizable interaction Neural Network by
    `Schütt et al. <https://proceedings.mlr.press/v139/schutt21a.html>`_.

    In order to accomodate general inputs/outputs, this PaiNN is different from the
    original in a few ways; the main change is that inputs vectors are not initialized
    to 0 anymore but to the time average of velocity.

    .. image:: https://i.imgur.com/NxZ2rPi.png

    """

    scalar_emb: LinearXav
    vector_emb: LinearXav
    filter_net: LinearXav
    layers: List[PaiNNLayer]
    readout: _PaiNNReadout
    radial_basis_fn: eqx.Module
    cutoff_fn: Optional[eqx.Module]

    hidden_size: int = eqx.field(static=True)
    output_size: int = eqx.field(static=True)
    num_mp_steps: int = eqx.field(static=True)
    n_vels: int = eqx.field(static=True)
    homogeneous_particles: bool = eqx.field(static=True)
    shared_filters: bool = eqx.field(static=True)
    shared_interactions: bool = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        num_mp_steps: int,
        radial_basis_fn: eqx.Module,
        cutoff_fn: Optional[eqx.Module],
        n_vels: int,
        n_rbf: int,
        s_in_size: int,
        v_in_channels: int,
        homogeneous_particles: bool = True,
        activation: Callable = jax.nn.silu,
        shared_interactions: bool = False,
        shared_filters: bool = False,
        eps: float = 1e-8,
        *,
        key,
    ):
        """Initialize the model.

        Args:
            hidden_size: Determines the size of each embedding vector.
            output_size: Number of output features.
            num_mp_steps: Number of interaction blocks.
            radial_basis_fn: Expands inter-particle distances in a basis set.
            cutoff_fn: Cutoff function.
            n_vels: Number of historical velocities.
            n_rbf: Number of radial basis functions (input size of ``filter_net``).
            s_in_size: Size of the input scalar features (vel_mag + optional
                particle-type one-hot).
            v_in_channels: Number of input vector channels (trajectory + force +
                bound depending on the dataset).
            homogeneous_particles: If all particles are of homogeneous type.
            activation: Activation function.
            shared_interactions: If True, share the weights across interaction blocks.
            shared_filters: If True, share the weights across filter networks.
            eps: Constant added in norm to prevent derivation instabilities.
            key: PRNG key for initialization.
        """
        assert radial_basis_fn is not None, "A radial_basis_fn must be provided"

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_mp_steps = num_mp_steps
        self.n_vels = n_vels
        self.homogeneous_particles = homogeneous_particles
        self.shared_filters = shared_filters
        self.shared_interactions = shared_interactions
        self.eps = eps
        self.radial_basis_fn = radial_basis_fn
        self.cutoff_fn = cutoff_fn

        k_s, k_v, k_f, k_l, k_r = jax.random.split(key, 5)
        self.scalar_emb = LinearXav(s_in_size, hidden_size, key=k_s)
        self.vector_emb = LinearXav(
            v_in_channels, hidden_size, with_bias=False, key=k_v
        )
        filter_out = (
            3 * hidden_size if shared_filters else num_mp_steps * 3 * hidden_size
        )
        self.filter_net = LinearXav(n_rbf, filter_out, key=k_f)

        keys_layers = jax.random.split(k_l, num_mp_steps)
        if shared_interactions:
            layer0 = PaiNNLayer(hidden_size, activation=activation, eps=eps, key=keys_layers[0])
            self.layers = [layer0] * num_mp_steps
        else:
            self.layers = [
                PaiNNLayer(hidden_size, activation=activation, eps=eps, key=keys_layers[i])
                for i in range(num_mp_steps)
            ]

        self.readout = _PaiNNReadout(
            hidden_size=hidden_size,
            in_channels=hidden_size,
            out_channels=output_size,
            activation=activation,
            blocks=2,
            eps=eps,
            key=k_r,
        )

    def _embed(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """Embed the input nodes."""
        s = jnp.asarray(graph.nodes.s, dtype=jnp.float32)
        if len(s.shape) == 1:
            s = s[:, jnp.newaxis]
        s = self.scalar_emb(s)[:, jnp.newaxis]
        v = self.vector_emb(graph.nodes.v)
        return graph._replace(nodes=NodeFeatures(s=s, v=v))

    def _get_filters(self, norm_ij: jnp.ndarray) -> List[jnp.ndarray]:
        r"""Compute the rotationally invariant filters :math:`W_s`.

        .. math::
            W_s = MLP(RBF(\|\vector{r}_{ij}\|)) * f_{cut}(\|\vector{r}_{ij}\|)
        """
        phi_ij = self.radial_basis_fn(norm_ij)
        if self.cutoff_fn is not None:
            norm_ij = self.cutoff_fn(norm_ij)
        filters = self.filter_net(phi_ij) * norm_ij[:, jnp.newaxis]
        if self.shared_filters:
            filter_list = [filters] * self.num_mp_steps
        else:
            filter_list = jnp.split(filters, self.num_mp_steps, axis=-1)
        return filter_list

    def _transform(
        self, features: Dict[str, jnp.ndarray], particle_type: jnp.ndarray
    ) -> jraph.GraphsTuple:
        n_nodes = particle_type.shape[0]

        node_scalars = []
        node_vectors = []
        traj = jnp.reshape(features["vel_hist"], (n_nodes, self.n_vels, -1))
        node_vectors.append(traj.transpose(0, 2, 1))
        if "force" in features:
            node_vectors.append(features["force"][..., jnp.newaxis])
        if "bound" in features:
            bounds = jnp.reshape(features["bound"], (n_nodes, 2, -1))
            node_vectors.append(bounds.transpose(0, 2, 1))
        node_scalars.append(features["vel_mag"])
        if not self.homogeneous_particles:
            particles = jax.nn.one_hot(particle_type, NodeType.SIZE)
            node_scalars.append(particles)

        node_scalars = jnp.concatenate(node_scalars, axis=-1)
        node_vectors = jnp.concatenate(node_vectors, axis=-1)

        return jraph.GraphsTuple(
            nodes=NodeFeatures(s=node_scalars, v=node_vectors),
            edges=features["rel_disp"],
            senders=features["senders"],
            receivers=features["receivers"],
            n_node=jnp.array([n_nodes]),
            n_edge=jnp.array([len(features["senders"])]),
            globals=None,
        )

    def __call__(
        self, sample: Tuple[Dict[str, jnp.ndarray], jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        graph = self._transform(*sample)
        # compute atom and pair features
        norm_ij = jnp.sqrt(jnp.sum(graph.edges**2, axis=1, keepdims=True) + self.eps)
        # edge directions
        dir_ij = graph.edges / (norm_ij + self.eps)
        graph = graph._replace(edges=dir_ij)

        filter_list = self._get_filters(norm_ij)

        graph = self._embed(graph)
        for n, layer in enumerate(self.layers):
            graph = layer(graph, filter_list[n])

        _, v = self.readout(graph)
        return {"acc": v}
