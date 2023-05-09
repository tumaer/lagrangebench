import warnings
from collections import defaultdict
from functools import partial
from typing import Callable, Dict, List

import jax
import jax.numpy as jnp
import numpy as np

MetricsDict = Dict[str, Dict[str, jnp.ndarray]]


class MetricsComputer:
    """Metrics between predicted and target rollouts.
    Args:
        active_metrics: list of metrics to compute
        dist: distance function
        metadata: metadata of the dataset
        stride: stride for computing metrics
    """

    METRICS = ["mse", "mae", "sinkhorn", "emd", "e_kin"]

    def __init__(
        self,
        active_metrics: List,
        dist: Callable,
        metadata: Dict,
        input_seq_length: int,
        stride: int = 10,
    ):
        assert all([hasattr(self, metric) for metric in active_metrics])
        self._active_metrics = active_metrics
        self._dist = dist
        self._dist_vmap = jax.vmap(dist, in_axes=(0, 0))
        self._dist_dvmap = jax.vmap(self._dist_vmap, in_axes=(0, 0))
        self._input_seq_length = input_seq_length
        self._stride = stride
        self._metadata = metadata

    def __call__(
        self, pred_rollout: jnp.ndarray, target_rollout: jnp.ndarray
    ) -> MetricsDict:
        # both pred_rollout and target_rollout are of shape
        # (traj_len-input_seq_length, num_particles, dim)
        metrics = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for metric_name in self._active_metrics:
                metric_fn = getattr(self, metric_name)
                if metric_name in ["mse", "mae"]:
                    metrics[metric_name] = jax.vmap(metric_fn)(
                        pred_rollout, target_rollout
                    )

                    for span in [5, 10, 20, 50, 100]:
                        metrics[metric_name + str(span)] = metrics[metric_name][:span]

                elif metric_name in ["e_kin"]:
                    dt = self._metadata["dt"]
                    dx = self._metadata["dx"]
                    dim = self._metadata["dim"]

                    metric_dvmap = jax.vmap(jax.vmap(metric_fn))

                    # Ekin of predicted rollout
                    velocity_rollout = self._dist_dvmap(
                        pred_rollout[1 :: self._stride],
                        pred_rollout[0 : -1 : self._stride],
                    )
                    e_kin_pred = metric_dvmap(velocity_rollout / dt).sum(1)

                    # Ekin of target rollout
                    velocity_rollout = self._dist_dvmap(
                        target_rollout[1 :: self._stride],
                        target_rollout[0 : -1 : self._stride],
                    )
                    e_kin_target = metric_dvmap(velocity_rollout / dt).sum(1)

                    metrics[metric_name] = {
                        "metadata": self._metadata,
                        "predicted": e_kin_pred * dx**dim,
                        "target": e_kin_target * dx**dim,
                    }

                elif metric_name in ["sinkhorn", "emd"]:
                    # vmap blows up for emd and sinkhorn (distance matrix)
                    _, metrics[metric_name] = jax.lax.scan(
                        lambda _, x: (None, metric_fn(*x)),
                        None,
                        (
                            pred_rollout[0 : -1 : self._stride],
                            target_rollout[0 : -1 : self._stride],
                        ),
                    )
        return metrics

    @partial(jax.jit, static_argnums=(0,))
    def mse(self, pred: jnp.ndarray, target: jnp.ndarray):
        return (self._dist_vmap(pred, target) ** 2).mean()

    @partial(jax.jit, static_argnums=(0,))
    def mae(self, pred: jnp.ndarray, target: jnp.ndarray):
        return (jnp.abs(self._dist_vmap(pred, target))).mean()

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
        def emd2_(a, b, loss_matrix):
            return jnp.array(emd2(a, b, loss_matrix, numItermax=50000))

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
        def sinkhorn2_(a, b, loss_matrix):
            return jnp.array(
                sinkhorn2(a, b, loss_matrix, reg=0.1, numItermax=500, stopThr=1e-05)
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

        def dist(a, b):
            return jnp.sum(self._dist(a, b) ** 2)

        if not squared:

            def dist(a, b):
                return jnp.sqrt(dist(a, b))

        return jnp.array(jax.vmap(lambda a: jax.vmap(lambda b: dist(a, b))(y))(x))

    @partial(jax.jit, static_argnums=(0,))
    def e_kin(self, frame: jnp.ndarray):
        """Computes the kinetic energy of a frame"""
        return jnp.sum(frame**2)  # * dx ** 3


def averaged_metrics(eval_metrics: MetricsDict) -> Dict[str, float]:
    """Averages the metrics over the rollouts."""
    # create a dictionary with the same keys as the metrics, but empty list as values
    trajectory_averages = defaultdict(list)
    for rollout in eval_metrics.values():
        for k, v in rollout.items():
            if k == "e_kin":
                continue
            if k in ["mse", "mae"]:
                k = "loss"
            trajectory_averages[k].append(jnp.mean(v).item())

    # mean and std values accross rollouts
    small_metrics = {}
    for k, v in trajectory_averages.items():
        small_metrics[f"val/{k}"] = float(np.mean(v))
    for k, v in trajectory_averages.items():
        small_metrics[f"val/std{k}"] = float(np.std(v))

    return small_metrics
