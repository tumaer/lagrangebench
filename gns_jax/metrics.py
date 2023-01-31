import warnings
from functools import partial
from typing import Callable, Dict, List

import jax
import jax.numpy as jnp


class MetricsComputer:
    """Metrics between predicted and target rollouts."""

    # TODO for now
    METRICS = ["mse", "mae", "sinkhorn", "emd"]

    def __init__(
        self,
        active_metrics: List,
        dist: Callable,
        divergence_step: int = 10,
    ):
        assert all([hasattr(self, metric) for metric in active_metrics])
        self._active_metrics = active_metrics
        self._dist = dist
        self._dist_vmap = jax.vmap(dist, in_axes=(0, 0))
        self._divergence_step = divergence_step

    def __call__(self, pred_rollout: jnp.ndarray, target_rollout: jnp.ndarray) -> Dict:
        assert pred_rollout.shape[0] == target_rollout.shape[0]
        metrics = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for metric_name in self._active_metrics:
                metric_fn = getattr(self, metric_name)
                if metric_name in ["mse", "mae"]:
                    metrics[metric_name] = jax.vmap(metric_fn)(
                        pred_rollout, target_rollout
                    )
                else:
                    # vmap blows up for emd and sinkhorn (distance matrix)
                    _, metrics[metric_name] = jax.lax.scan(
                        lambda _, x: (None, metric_fn(*x)),
                        None,
                        (
                            pred_rollout[0 : -1 : self._divergence_step],
                            target_rollout[0 : -1 : self._divergence_step],
                        ),
                    )
        return metrics

    @partial(jax.jit, static_argnums=(0,))
    def mse(self, pred: jnp.ndarray, target: jnp.ndarray):
        return (self._dist_vmap(pred, target) ** 2).mean(1, 2)

    @partial(jax.jit, static_argnums=(0,))
    def mae(self, pred: jnp.ndarray, target: jnp.ndarray):
        return (jnp.abs(self._dist_vmap(pred, target))).mean(1, 2)

    @partial(jax.jit, static_argnums=(0,))
    def sinkhorn(self, pred: jnp.ndarray, target: jnp.ndarray):
        # equivalent to empirical_sinkhorn_divergence with custom distance computation
        sinkhorn_ab = self._custom_empirical_sinkorn2(pred, target)
        sinkhorn_a = self._custom_empirical_sinkorn2(pred, pred)
        sinkhorn_b = self._custom_empirical_sinkorn2(target, target)
        return jnp.clip(sinkhorn_ab - 0.5 * (sinkhorn_a + sinkhorn_b), 0)

    @partial(jax.jit, static_argnums=(0,))
    def emd(self, pred: jnp.ndarray, target: jnp.ndarray):
        from ot import emd2

        loss_matrix = self._distance_matrix(pred, target)
        # weights are uniform
        a, b = (
            jnp.ones((pred.shape[0],)) / pred.shape[0],
            jnp.ones((target.shape[0],)) / target.shape[0],
        )
        shape = jax.ShapeDtypeStruct((), dtype=jnp.float32)
        # hack to avoid CpuCallback attribute error
        emd2_ = lambda a, b, loss_matrix: jnp.array(
            emd2(a, b, loss_matrix, numItermax=50000)
        )
        return jax.pure_callback(emd2_, shape, a, b, loss_matrix)

    @partial(jax.jit, static_argnums=(0,))
    def _custom_empirical_sinkorn2(self, pred: jnp.ndarray, target: jnp.ndarray):
        from ot.bregman import sinkhorn2

        # weights are uniform
        a, b = (
            jnp.ones((pred.shape[0],)) / pred.shape[0],
            jnp.ones((target.shape[0],)) / target.shape[0],
        )
        loss_matrix = self._distance_matrix(pred, target)
        shape = jax.ShapeDtypeStruct((), dtype=jnp.float32)
        # hack to avoid CpuCallback attribute error
        sinkhorn2_ = lambda a, b, loss_matrix: jnp.array(
            sinkhorn2(a, b, loss_matrix, reg=0.1, numItermax=500, stopThr=1e-5)
        )
        return jax.pure_callback(
            sinkhorn2_,
            shape,
            a,
            b,
            loss_matrix,
        )

    def _distance_matrix(
        self, x: jnp.ndarray, y: jnp.ndarray, squared=True
    ) -> jnp.ndarray:
        """Euclidean distance matrix."""
        # TODO maybe distances have to be rescaled/normalized
        dist = lambda a, b: jnp.sum(self._dist(a, b) ** 2)
        if not squared:
            dist = lambda a, b: jnp.sqrt(dist(a, b))
        return jnp.array(jax.vmap(lambda a: jax.vmap(lambda b: dist(a, b))(y))(x))
