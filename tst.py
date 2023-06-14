import jax
import jax.numpy as jnp

if __name__ == "__main__":

    def f(A, x):
        return jnp.tanh(jnp.dot(A, x)), False

    # f = lambda A, x: jnp.tanh(jnp.dot(A, x))
    A = jax.ShapeDtypeStruct((2000, 3000), jnp.float32)
    x = jax.ShapeDtypeStruct((3000, 1000), jnp.float32)
    out = jax.eval_shape(f, A, x)  # no FLOPs performed
    print(out.shape)
