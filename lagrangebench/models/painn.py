"""
Modified PaiNN implementation for general vectorial inputs and outputs
`Schütt et al. <https://proceedings.mlr.press/v139/schutt21a.html>`_.
PaiNN model, layers and feature transform.

Original implementation: https://github.com/atomistic-machine-learning/schnetpack

Standalone implementation + validation: https://github.com/gerkone/painn-jax
"""

from typing import Callable, Dict, NamedTuple, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jax.tree_util as tree
import jraph

from lagrangebench.utils import NodeType

from .utils import LinearXav


class NodeFeatures(NamedTuple):
    """Simple container for PaiNN scalar and vectorial node features."""

    s: jnp.ndarray = None
    v: jnp.ndarray = None


ReadoutFn = Callable[[jraph.GraphsTuple], Tuple[jnp.ndarray, jnp.ndarray]]
ReadoutBuilderFn = Callable[..., ReadoutFn]


class GatedEquivariantBlock(hk.Module):
    """Gated equivariant block (restricted to vectorial features).

    .. image:: https://i.imgur.com/EMlg2Qi.png
    """

    def __init__(
        self,
        hidden_size: int,
        scalar_out_channels: int,
        vector_out_channels: int,
        activation: Callable = jax.nn.silu,
        scalar_activation: Callable = None,
        eps: float = 1e-8,
        name: str = "gated_equivariant_block",
    ):
        """Initialize the layer.

        Args:
            hidden_size: Number of hidden channels.
            scalar_out_channels: Number of scalar output channels.
            vector_out_channels: Number of vector output channels.
            activation: Gate activation function.
            scalar_activation: Activation function for the scalar output.
            eps: Constant added in norm to prevent derivation instabilities.
            name: Name of the module.

        """
        super().__init__(name)

        assert scalar_out_channels > 0 and vector_out_channels > 0
        self._scalar_out_channels = scalar_out_channels
        self._vector_out_channels = vector_out_channels
        self._eps = eps

        self.vector_mix_net = LinearXav(
            2 * vector_out_channels,
            with_bias=False,
            name="vector_mix_net",
        )
        self.gate_block = hk.Sequential(
            [
                LinearXav(hidden_size),
                activation,
                LinearXav(scalar_out_channels + vector_out_channels),
            ],
            name="scalar_gate_net",
        )
        self.scalar_activation = scalar_activation

    def __call__(
        self, s: jnp.ndarray, v: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        v_l, v_r = jnp.split(self.vector_mix_net(v), 2, axis=-1)

        v_r_norm = jnp.sqrt(jnp.sum(v_r**2, axis=-2) + self._eps)
        gating_scalars = jnp.concatenate([s, v_r_norm], axis=-1)
        s, _, v_gate = jnp.split(
            self.gate_block(gating_scalars),
            [self._scalar_out_channels, self._vector_out_channels],
            axis=-1,
        )
        # scale the vectors by the gating scalars
        v = v_l * v_gate[:, jnp.newaxis]

        if self.scalar_activation:
            s = self.scalar_activation(s)

        return s, v


def gaussian_rbf(
    n_rbf: int,
    cutoff: float,
    start: float = 0.0,
    centered: bool = False,
    trainable: bool = False,
) -> Callable[[jnp.ndarray], Callable]:
    r"""Gaussian radial basis functions.

    Args:
        n_rbf: total number of Gaussian functions, :math:`N_g`.
        cutoff: center of last Gaussian function, :math:`\mu_{N_g}`
        start: center of first Gaussian function, :math:`\mu_0`.
        trainable: If True, widths and offset of Gaussian functions learnable.
    """
    if centered:
        widths = jnp.linspace(start, cutoff, n_rbf)
        offset = jnp.zeros_like(widths)
    else:
        offset = jnp.linspace(start, cutoff, n_rbf)
        width = jnp.abs(cutoff - start) / n_rbf * jnp.ones_like(offset)

    if trainable:
        widths = hk.get_parameter(
            "widths", width.shape, width.dtype, init=lambda *_: width
        )
        offsets = hk.get_parameter(
            "offset", offset.shape, offset.dtype, init=lambda *_: offset
        )
    else:
        hk.set_state("widths", jnp.array([width]))
        hk.set_state("offsets", jnp.array([offset]))
        widths = hk.get_state("widths")
        offsets = hk.get_state("offsets")

    def _rbf(x: jnp.ndarray) -> jnp.ndarray:
        coeff = -0.5 / jnp.power(widths, 2)
        diff = x[..., jnp.newaxis] - offsets
        return jnp.exp(coeff * jnp.power(diff, 2))

    return _rbf


def cosine_cutoff(cutoff: float) -> Callable[[jnp.ndarray], Callable]:
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
    hk.set_state("cutoff", cutoff)
    cutoff = hk.get_state("cutoff")

    def _cutoff(x: jnp.ndarray) -> jnp.ndarray:
        # Compute values of cutoff function
        cuts = 0.5 * (jnp.cos(x * jnp.pi / cutoff) + 1.0)
        # Remove contributions beyond the cutoff radius
        mask = jnp.array(x < cutoff, dtype=jnp.float32)
        return cuts * mask

    return _cutoff


def PaiNNReadout(
    hidden_size: int,
    out_channels: int = 1,
    activation: Callable = jax.nn.silu,
    blocks: int = 2,
    eps: float = 1e-8,
) -> ReadoutFn:
    """
    PaiNN readout block.

    Args:
        hidden_size: Number of hidden channels.
        scalar_out_channels: Number of scalar/vector output channels.
        activation: Activation function.
        blocks: Number of readout blocks.

    Returns:
        Configured readout function.
    """

    def _readout(graph: jraph.GraphsTuple) -> Tuple[jnp.ndarray, jnp.ndarray]:
        s, v = graph.nodes
        s = jnp.squeeze(s)
        for i in range(blocks - 1):
            ith_hidden_size = hidden_size // 2 ** (i + 1)
            s, v = GatedEquivariantBlock(
                hidden_size=ith_hidden_size * 2,
                scalar_out_channels=ith_hidden_size,
                vector_out_channels=ith_hidden_size,
                activation=activation,
                eps=eps,
                name=f"readout_block_{i}",
            )(s, v)

        s, v = GatedEquivariantBlock(
            hidden_size=ith_hidden_size,
            scalar_out_channels=out_channels,
            vector_out_channels=out_channels,
            activation=activation,
            eps=eps,
            name="readout_block_out",
        )(s, v)

        return jnp.squeeze(s), jnp.squeeze(v)

    return _readout


class PaiNNLayer(hk.Module):
    """PaiNN interaction block."""

    def __init__(
        self,
        hidden_size: int,
        layer_num: int,
        activation: Callable = jax.nn.silu,
        blocks: int = 2,
        aggregate_fn: Callable = jraph.segment_sum,
        eps: float = 1e-8,
    ):
        """
        Initialize the PaiNN layer, made up of an interaction block and a mixing block.

        Args:
            hidden_size: Number of node features.
            activation: Activation function.
            layer_num: Numbering of the layer.
            blocks: Number of layers in the context networks.
            aggregate_fn: Function to aggregate the neighbors.
            eps: Constant added in norm to prevent derivation instabilities.
        """
        super().__init__(f"layer_{layer_num}")
        self._hidden_size = hidden_size
        self._eps = eps
        self._aggregate_fn = aggregate_fn

        # inter-particle context net
        self.interaction_block = hk.Sequential(
            [LinearXav(hidden_size), activation] * (blocks - 1)
            + [LinearXav(3 * hidden_size)],
            name="interaction_block",
        )

        # intra-particle context net
        self.mixing_block = hk.Sequential(
            [LinearXav(hidden_size), activation] * (blocks - 1)
            + [LinearXav(3 * hidden_size)],
            name="mixing_block",
        )

        # vector channel mix
        self.vector_mixing_block = LinearXav(
            2 * hidden_size,
            with_bias=False,
            name="vector_mixing_block",
        )

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
        x = self.interaction_block(s)

        xj = x[receivers]
        vj = v[receivers]

        ds, dv1, dv2 = jnp.split(Wij * xj, 3, axis=-1)
        n_nodes = tree.tree_leaves(s)[0].shape[0]
        dv = dv1 * dir_ij[..., jnp.newaxis] + dv2 * vj
        # aggregate scalars and vectors
        ds = self._aggregate_fn(ds, senders, n_nodes)
        dv = self._aggregate_fn(dv, senders, n_nodes)

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
        v_norm = jnp.sqrt(jnp.sum(v_r**2, axis=-2, keepdims=True) + self._eps)

        ts = jnp.concatenate([s, v_norm], axis=-1)
        ds, dv, dsv = jnp.split(self.mixing_block(ts), 3, axis=-1)
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


class PaiNN(hk.Module):
    r"""Polarizable interaction Neural Network by
    `Schütt et al. <https://proceedings.mlr.press/v139/schutt21a.html>`_.

    In order to accomodate general inputs/outputs, this PaiNN is different from the
    original in a few ways; the main change is that inputs vectors are not initialized
    to 0 anymore but to the time average of velocity.

    .. image:: https://i.imgur.com/NxZ2rPi.png

    """

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        num_mp_steps: int,
        radial_basis_fn: Callable,
        cutoff_fn: Callable,
        n_vels: int,
        homogeneous_particles: bool = True,
        activation: Callable = jax.nn.silu,
        shared_interactions: bool = False,
        shared_filters: bool = False,
        eps: float = 1e-8,
    ):
        """Initialize the model.

        Args:
            hidden_size: Determines the size of each embedding vector.
            output_size: Number of output features.
            num_mp_steps: Number of interaction blocks.
            radial_basis_fn: Expands inter-particle distances in a basis set.
            cutoff_fn: Cutoff function.
            n_vels: Number of historical velocities.
            homogeneous_particles: If all particles are of homogeneous type.
            activation: Activation function.
            shared_interactions: If True, share the weights across interaction blocks.
            shared_filters: If True, share the weights across filter networks.
            eps: Constant added in norm to prevent derivation instabilities.
        """
        super().__init__("painn")

        assert radial_basis_fn is not None, "A radial_basis_fn must be provided"

        self._n_vels = n_vels
        self._homogeneous_particles = homogeneous_particles
        self._hidden_size = hidden_size
        self._num_mp_steps = num_mp_steps
        self._eps = eps
        self._shared_filters = shared_filters
        self._shared_interactions = shared_interactions

        self.radial_basis_fn = radial_basis_fn
        self.cutoff_fn = cutoff_fn

        self.scalar_emb = LinearXav(self._hidden_size, name="scalar_embedding")
        # mix vector channels (only used if vector features are present in input)
        self.vector_emb = LinearXav(
            self._hidden_size, with_bias=False, name="vector_embedding"
        )

        if shared_filters:
            self.filter_net = LinearXav(3 * self._hidden_size, name="filter_net")
        else:
            self.filter_net = LinearXav(
                self._num_mp_steps * 3 * self._hidden_size, name="filter_net"
            )

        if self._shared_interactions:
            self.layers = [
                PaiNNLayer(self._hidden_size, 0, activation, eps=eps)
            ] * self._num_mp_steps
        else:
            self.layers = [
                PaiNNLayer(self._hidden_size, i, activation, eps=eps)
                for i in range(self._num_mp_steps)
            ]

        self._readout = PaiNNReadout(self._hidden_size, out_channels=output_size)

    def _embed(self, graph: jraph.GraphsTuple) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Embed the input nodes."""
        # embeds scalar features
        s = jnp.asarray(graph.nodes.s, dtype=jnp.float32)
        if len(s.shape) == 1:
            s = s[:, jnp.newaxis]
        s = self.scalar_emb(s)[:, jnp.newaxis]

        # embeds vector features
        v = self.vector_emb(graph.nodes.v)

        return graph._replace(nodes=NodeFeatures(s=s, v=v))

    def _get_filters(self, norm_ij: jnp.ndarray) -> jnp.ndarray:
        r"""Compute the rotationally invariant filters :math:`W_s`.

        .. math::
            W_s = MLP(RBF(\|\vector{r}_{ij}\|)) * f_{cut}(\|\vector{r}_{ij}\|)
        """
        phi_ij = self.radial_basis_fn(norm_ij)
        if self.cutoff_fn is not None:
            norm_ij = self.cutoff_fn(norm_ij)
        # compute filters
        filters = self.filter_net(phi_ij) * norm_ij[:, jnp.newaxis]
        # split into layer-wise filters
        if self._shared_filters:
            filter_list = [filters] * self._num_mp_steps
        else:
            filter_list = jnp.split(filters, self._num_mp_steps, axis=-1)
        return filter_list

    def _transform(
        self, features: Dict[str, jnp.ndarray], particle_type: jnp.ndarray
    ) -> jraph.GraphsTuple:
        n_nodes = particle_type.shape[0]

        # node features
        node_scalars = []
        node_vectors = []
        traj = jnp.reshape(features["vel_hist"], (n_nodes, self._n_vels, -1))
        node_vectors.append(traj.transpose(0, 2, 1))
        if "force" in features:
            node_vectors.append(features["force"][..., jnp.newaxis])
        if "bound" in features:
            bounds = jnp.reshape(features["bound"], (n_nodes, 2, -1))
            node_vectors.append(bounds.transpose(0, 2, 1))
        # velocity magnitudes as node feature
        node_scalars.append(features["vel_mag"])
        if not self._homogeneous_particles:
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
        norm_ij = jnp.sqrt(jnp.sum(graph.edges**2, axis=1, keepdims=True) + self._eps)
        # edge directions
        dir_ij = graph.edges / (norm_ij + self._eps)
        graph = graph._replace(edges=dir_ij)

        # compute filters (r_ij track in message block from the paper)
        filter_list = self._get_filters(norm_ij)

        # embeds node scalar features (and vector, if present)
        graph = self._embed(graph)

        # message passing
        for n, layer in enumerate(self.layers):
            graph = layer(graph, filter_list[n])

        _, v = self._readout(graph)
        return {"acc": v}
