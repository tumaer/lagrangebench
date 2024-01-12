import jax.numpy as jnp
import math

def fourier_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings.

    Args:
        timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
        dim (int): the dimension of the output.
        max_period (int): controls the minimum frequency of the embeddings.

    Returns:
        embedding (jax.numpy.DeviceArray): [N x dim] array of positional embeddings.
    """
    half = dim // 2
    freqs = jnp.exp(-math.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half)
    args = timesteps[:, None].astype(jnp.float32) * freqs[None, :]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2: #enters the 'if-statement' if the dim is odd
        embedding = jnp.concatenate([embedding, jnp.zeros((embedding.shape[0], 1), dtype=embedding.dtype)], axis=-1)
    return embedding

# features = jnp.ones((128,)) #3200 for RPF_2d
# print(features.shape)
# fourier_embedding(features, 64)