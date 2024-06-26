"""Training tricks and strategies, currently: random-walk noise and push forward."""

from typing import Tuple

import jax
import jax.numpy as jnp
from jax_sph.jax_md.partition import space

from lagrangebench.utils import get_kinematic_mask


def add_gns_noise(
    key: jax.Array,
    pos_input: jnp.ndarray,
    particle_type: jnp.ndarray,
    input_seq_length: int,
    noise_std: float,
    shift_fn: space.ShiftFn,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    r"""GNS-like random walk noise injection as described by
    `Sanchez-Gonzalez et al. <https://arxiv.org/abs/2002.09405>`_.

    Applies random-walk noise to the input positions and adjusts the targets accordingly
    to keep the trajectory consistent. It works by drawing independent samples from
    :math:`\mathcal{N^{(t)}}(0, \sigma_v^{(t)})` for each input state. Noise is
    accummulated as a random walk and added to the velocity seqence.
    Each :math:`\sigma_v^{(t)}` is set so that the last step of the random walk has
    :math:`\sigma_v^{(input\_seq\_length)}=noise\_std`. Based on the noised velocities,
    positions are adjusted such that :math:`\dot{p}^{t_k} = p^{t_k} − p^{t_{k−1}}`.

    Args:
        key: Random key.
        pos_input: Clean input positions. Shape:
            (num_particles_max, input_seq_length + pushforward["unrolls"][-1] + 1, dim)
        particle_type: Particle type vector. Shape: (num_particles_max,)
        input_seq_length: Input sequence length, as in the configs.
        noise_std: Noise standard deviation at the last sequence step.
        shift_fn: Shift function.
    """
    isl = input_seq_length
    # random-walk noise in the velocity applied to the first input_seq_length positions
    key, pos_input_noise = _get_random_walk_noise_for_pos_sequence(
        key, pos_input[:, :input_seq_length], noise_std_last_step=noise_std
    )

    kinematic_mask = get_kinematic_mask(particle_type)
    pos_input_noise = jnp.where(kinematic_mask[:, None, None], 0.0, pos_input_noise)
    # adjust targets based on the noise from the last input position
    n_potential_targets = pos_input[:, isl:].shape[1]
    pos_target_noise = pos_input_noise[:, -1][:, None, :]
    pos_target_noise = jnp.tile(pos_target_noise, (1, n_potential_targets, 1))
    pos_input_noise = jnp.concatenate([pos_input_noise, pos_target_noise], axis=1)

    shift_vmap = jax.vmap(shift_fn, in_axes=(0, 0))
    shift_dvmap = jax.vmap(shift_vmap, in_axes=(0, 0))
    pos_input_noisy = shift_dvmap(pos_input, pos_input_noise)

    return key, pos_input_noisy


def _get_random_walk_noise_for_pos_sequence(
    key, position_sequence, noise_std_last_step
):
    key, subkey = jax.random.split(key)
    velocity_sequence_shape = list(position_sequence.shape)
    velocity_sequence_shape[1] -= 1
    n_velocities = velocity_sequence_shape[1]

    velocity_sequence_noise = jax.random.normal(
        subkey, shape=tuple(velocity_sequence_shape)
    )
    velocity_sequence_noise *= noise_std_last_step / (n_velocities**0.5)
    velocity_sequence_noise = jnp.cumsum(velocity_sequence_noise, axis=1)

    position_sequence_noise = jnp.concatenate(
        [
            jnp.zeros_like(velocity_sequence_noise[:, 0:1]),
            jnp.cumsum(velocity_sequence_noise, axis=1),
        ],
        axis=1,
    )

    return key, position_sequence_noise


def push_forward_sample_steps(key, step, pushforward):
    """Sample the number of unroll steps based on the current training step and the
    specified pushforward configuration.

    Args:
        key: Random key
        step: Current training step
        pushforward: Pushforward configuration
    """
    key, key_unroll = jax.random.split(key, 2)

    # steps needs to be an ordered list
    steps = jnp.array(pushforward.steps)
    assert all(steps[i] <= steps[i + 1] for i in range(len(steps) - 1))

    # until which index to sample from
    idx = (step > steps).sum()

    unroll_steps = jax.random.choice(
        key_unroll,
        a=jnp.array(pushforward.unrolls[:idx]),
        p=jnp.array(pushforward.probs[:idx]),
    )
    return key, unroll_steps


def push_forward_build(model_apply, case):
    r"""Build the push forward function, introduced by
    `Brandstetter et al. <https://arxiv.org/abs/2202.03376>`_.

    Pushforward works by adding a stability "pushforward" loss term, in the form of an
    adversarial style loss.

    .. math::
        L_{pf} = \mathbb{E}_k \mathbb{E}_{u^{k+1} | u^k}
            \mathbb{E}_{\epsilon} \left[ \mathcal{L}(f(u^k + \epsilon), u^{k-1}) \right]

    where :math:`\epsilon` is :math:`u^k + \epsilon = f(u^{k−1})`, i.e. the 2-step
    unroll of the solver :math:`f` (from step :math:`k-1` to :math:`k`).
    The total loss is then :math:`L_{total}=\mathcal{L}(f(u^k), u^{k-1}) + L_{pf}`.
    Similarly, for :math:`S > 2` pushforward steps, :math:`L_{pf}` is extended to
    :math:`u^{k-S} \dots u^{k-1}` with cumulated :math:`\epsilon` perturbations.

    In practice, this is implemented by unrolling the solver for two steps, but only
    running gradients through the last unroll step.

    Args:
        model_apply: Model apply function
        case: Case setup function
    """

    @jax.jit
    def push_forward_fn(features, current_pos, particle_type, neighbors, params, state):
        """Push forward function.

        Args:
            features: Input features
            current_pos: Current position
            particle_type: Particle type vector
            neighbors: Neighbor list
            params: Model parameters
            state: Model state
        """
        # no buffer overflow check here, since push forward acts on later epochs
        pred, _ = model_apply(params, state, (features, particle_type))
        next_pos = case.integrate(pred, current_pos)
        current_pos = jnp.concatenate(
            [current_pos[:, 1:], next_pos[:, None, :]], axis=1
        )

        features, neighbors = case.preprocess_eval(
            (current_pos, particle_type), neighbors
        )
        return current_pos, neighbors, features

    return push_forward_fn
