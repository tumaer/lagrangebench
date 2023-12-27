"""Data utils."""

from typing import Dict, List

import jax.numpy as jnp
import numpy as np


def get_dataset_stats(
    metadata: Dict[str, List[float]],
    is_isotropic_norm: bool,
    noise_std: float,
) -> Dict[str, Dict[str, jnp.ndarray]]:
    """Return the dataset statistics based on the metadata dictionary.

    Args:
        metadata: Dataset metadata dictionary.
        is_isotropic_norm:
            Whether to shift/scale dimensions equally instead of dimension-wise.
        noise_std: Standard deviation of the GNS-style noise.

    Returns:
        Dictionary with the dataset statistics.
    """
    acc_mean = jnp.array(metadata["acc_mean"])
    acc_std = jnp.array(metadata["acc_std"])
    vel_mean = jnp.array(metadata["vel_mean"])
    vel_std = jnp.array(metadata["vel_std"])

    if is_isotropic_norm:
        acc_mean = jnp.mean(acc_mean) * jnp.ones_like(acc_mean)
        acc_std = jnp.sqrt(jnp.mean(acc_std**2)) * jnp.ones_like(acc_std)
        vel_mean = jnp.mean(vel_mean) * jnp.ones_like(vel_mean)
        vel_std = jnp.sqrt(jnp.mean(vel_std**2)) * jnp.ones_like(vel_std)

    return {
        "acceleration": {
            "mean": acc_mean,
            "std": jnp.sqrt(acc_std**2 + noise_std**2),
        },
        "velocity": {
            "mean": vel_mean,
            "std": jnp.sqrt(vel_std**2 + noise_std**2),
        },
    }


def numpy_collate(batch) -> np.ndarray:
    """Collate helper for torch dataloaders."""
    # NOTE: to numpy to avoid copying twice (dataloader timeout).
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    if isinstance(batch[0], (tuple, list)):
        return type(batch[0])(numpy_collate(samples) for samples in zip(*batch))
    else:
        return np.asarray(batch)
