"""Training tricks and strategies, currently: random-walk noise and push forward."""

from typing import Tuple

import jax
import jax.numpy as jnp
from jax_md.partition import space

from lagrangebench.utils import get_kinematic_mask


def add_gns_noise(
    key: jax.random.KeyArray,
    pos_input: jnp.ndarray,
    particle_type: jnp.ndarray,
    input_seq_length: int,
    noise_std: float,
    shift_fn: space.ShiftFn,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """GNS-style random-walk noise injection on the input_seq_length trajectory.

    Args:
        key: Random key.
        pos_input: Clean input positions.
        particle_type: Particle type vector.
        input_seq_length: Input sequence length.
        noise_std: Noise standard deviation.
        shift_fn: Shift function.
    """
    isl = input_seq_length
    # add noise to the input and adjust the target accordingly
    key, pos_input_noise = _get_random_walk_noise_for_pos_sequence(
        key, pos_input, noise_std_last_step=noise_std
    )
    kinematic_mask = get_kinematic_mask(particle_type)
    pos_input_noise = jnp.where(kinematic_mask[:, None, None], 0.0, pos_input_noise)
    # adjust targets based on the noise from the last input position
    n_potential_targets = pos_input_noise[:, isl:].shape[1]
    pos_target_noise = pos_input_noise[:, isl - 1][:, None, :]
    pos_target_noise = jnp.tile(pos_target_noise, (1, n_potential_targets, 1))
    pos_input_noise = pos_input_noise.at[:, isl:].set(pos_target_noise)

    shift_vmap = jax.vmap(shift_fn, in_axes=(0, 0))
    shift_dvmap = jax.vmap(shift_vmap, in_axes=(0, 0))
    pos_input_noisy = shift_dvmap(pos_input, pos_input_noise)

    return key, pos_input_noisy


def _get_random_walk_noise_for_pos_sequence(
    key, position_sequence, noise_std_last_step
):
    """Return random-walk noise in the velocity applied to the position.

    Args:
        key: Random key.
        position_sequence: Position sequence.
        noise_std_last_step: Standard deviation of the noise at the last step.
    """
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
    """Sample the number of unroll steps based on the current training step.

    Args:
        key: Random key
        step: Current training step
        pushforward: Pushforward configuration
    """
    key, key_unroll = jax.random.split(key, 2)

    # steps needs to be an ordered list
    steps = jnp.array(pushforward["steps"])
    assert all(steps[i] <= steps[i + 1] for i in range(len(steps) - 1))

    # until which index to sample from
    idx = (step > steps).sum()

    unroll_steps = jax.random.choice(
        key_unroll,
        a=jnp.array(pushforward["unrolls"][:idx]),
        p=jnp.array(pushforward["probs"][:idx]),
    )
    return key, unroll_steps


def push_forward_build(model_apply, case):
    """Build the push forward function.

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
