"""Pytest fixtures and global test configuration."""

import jax

# Enable 64-bit precision for equivariance tests to pass within the
# strict tolerance used by ``ModelTest.assert_equivariant`` (1e-5). With
# float32 the accumulated numerical error from many message-passing
# aggregations can exceed this tolerance.
jax.config.update("jax_enable_x64", True)
