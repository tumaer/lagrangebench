"""Metrics for evaluation end testing."""

import warnings
from collections import defaultdict
from functools import partial
from typing import Callable, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
from ott.geometry.geometry import Geometry
from ott.tools.sinkhorn_divergence import sinkhorn_divergence

MetricsDict = Dict[str, Dict[str, jnp.ndarray]]


class MetricsComputer:
    """
    Metrics between predicted and target rollouts.

    Currently implemented:
    * MSE, mean squared error
    * MAE, mean absolute error
    * Sinkhorn distance, measures the similarity of two particle distributions
    * Kinetic energy, physical quantity of interest
    """

    METRICS = ["mse", "mae", "sinkhorn", "e_kin"]

    def __init__(
        self,
        active_metrics: List,
        dist_fn: Callable,
        metadata: Dict,
        input_seq_length: int,
        stride: int = 10,
        loss_ranges: Optional[List] = None,
        ot_backend: str = "ott",
    ):
        """Init the metric computer.

        Args:
            active_metrics: List of metrics to compute.
            dist_fn: Distance function.
            metadata: Metadata of the dataset.
            loss_ranges: List of horizon lengths to compute the loss for.
            input_seq_length: Length of the input sequence.
            stride: Rollout subsample frequency for e_kin and sinkhorn.
            ot_backend: Backend for sinkhorn computation. "ott" or "pot".
        """
        if active_metrics is None:
            active_metrics = []
        assert all([hasattr(self, metric) for metric in active_metrics])
        assert ot_backend in ["ott", "pot"]

        self._active_metrics = active_metrics
        self._dist_fn = dist_fn
        self._dist_vmap = jax.vmap(dist_fn, in_axes=(0, 0))
        self._dist_dvmap = jax.vmap(self._dist_vmap, in_axes=(0, 0))

        if loss_ranges is None:
            loss_ranges = [1, 5, 10, 20, 50, 100]
        self._loss_ranges = loss_ranges
        self._input_seq_length = input_seq_length
        self._stride = stride
        self._metadata = metadata
        self.ot_backend = ot_backend

    def __call__(
        self, pred_rollout: jnp.ndarray, target_rollout: jnp.ndarray
    ) -> MetricsDict:
        """Compute the metrics between two rollouts.

        Args:
            pred_rollout: Predicted rollout.
            target_rollout: Target rollout.

        Returns:
            Dictionary of metrics.
        """
        # both rollouts of shape (traj_len - t_window, n_nodes, dim)
        target_rollout = jnp.asarray(target_rollout, dtype=pred_rollout.dtype)
        metrics = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for metric_name in self._active_metrics:
                metric_fn = getattr(self, metric_name)
                if metric_name in ["mse", "mae"]:
                    # full rollout loss
                    metrics[metric_name] = jax.vmap(metric_fn)(
                        pred_rollout, target_rollout
                    )
                    # shorter horizon losses
                    for i in self._loss_ranges:
                        if i < metrics[metric_name].shape[0]:
                            metrics[f"{metric_name}{i}"] = metrics[metric_name][:i]

                elif metric_name in ["e_kin"]:
                    dt = self._metadata["dt"] * self._metadata["write_every"]
                    dx = self._metadata["dx"]
                    dim = self._metadata["dim"]

                    metric_dvmap = jax.vmap(jax.vmap(metric_fn))

                    # Ekin of predicted rollout
                    velocity_rollout = self._dist_dvmap(
                        pred_rollout[1 :: self._stride],
                        pred_rollout[0 : -1 : self._stride],
                    )
                    e_kin_pred = metric_dvmap(velocity_rollout / dt).sum(1)
                    e_kin_pred = e_kin_pred * dx**dim

                    # Ekin of target rollout
                    velocity_rollout = self._dist_dvmap(
                        target_rollout[1 :: self._stride],
                        target_rollout[0 : -1 : self._stride],
                    )
                    e_kin_target = metric_dvmap(velocity_rollout / dt).sum(1)
                    e_kin_target = e_kin_target * dx**dim

                    metrics[metric_name] = {
                        "predicted": e_kin_pred,
                        "target": e_kin_target,
                        "mse": ((e_kin_pred - e_kin_target) ** 2).mean(),
                    }

                elif metric_name == "sinkhorn":
                    # vmapping over distance matrix blows up
                    metrics[metric_name] = jax.lax.scan(
                        lambda _, x: (None, self.sinkhorn(*x)),
                        None,
                        (
                            pred_rollout[0 :: self._stride],
                            target_rollout[0 :: self._stride],
                        ),
                    )[1]
        return metrics

    @partial(jax.jit, static_argnums=(0,))
    def mse(self, pred: jnp.ndarray, target: jnp.ndarray) -> float:
        """Compute the mean squared error between two rollouts."""
        return (self._dist_vmap(pred, target) ** 2).mean()

    @partial(jax.jit, static_argnums=(0,))
    def mae(self, pred: jnp.ndarray, target: jnp.ndarray) -> float:
        """Compute the mean absolute error between two rollouts."""
        return (jnp.abs(self._dist_vmap(pred, target))).mean()

    @partial(jax.jit, static_argnums=(0,))
    def sinkhorn(self, pred: jnp.ndarray, target: jnp.ndarray) -> float:
        """Compute the sinkhorn distance between two rollouts."""
        if self.ot_backend == "ott":
            return self._sinkhorn_ott(pred, target)
        else:
            return self._sinkhorn_pot(pred, target)

    @partial(jax.jit, static_argnums=(0,))
    def e_kin(self, frame: jnp.ndarray) -> float:
        """Compute the kinetic energy of a frame."""
        return jnp.sum(frame**2)  # * dx ** 3

    def _sinkhorn_ott(self, pred: jnp.ndarray, target: jnp.ndarray) -> float:
        # pairwise distances as cost
        loss_matrix_xy = self._distance_matrix(pred, target)
        loss_matrix_xx = self._distance_matrix(pred, pred)
        loss_matrix_yy = self._distance_matrix(target, target)
        return sinkhorn_divergence(
            Geometry,
            loss_matrix_xy,
            loss_matrix_xx,
            loss_matrix_yy,
            # uniform weights
            a=jnp.ones((pred.shape[0],)) / pred.shape[0],
            b=jnp.ones((target.shape[0],)) / target.shape[0],
            sinkhorn_kwargs={"threshold": 1e-4},
        ).divergence

    def _sinkhorn_pot(self, pred: jnp.ndarray, target: jnp.ndarray):
        """Jax-compatible POT implementation of Sinkorn."""
        # equivalent to empirical_sinkhorn_divergence with custom distance computation
        sinkhorn_ab = self._custom_empirical_sinkorn_pot(pred, target)
        sinkhorn_a = self._custom_empirical_sinkorn_pot(pred, pred)
        sinkhorn_b = self._custom_empirical_sinkorn_pot(target, target)
        return jnp.asarray(
            jnp.clip(sinkhorn_ab - 0.5 * (sinkhorn_a + sinkhorn_b), 0),
            dtype=jnp.float32,
        )

    def _custom_empirical_sinkorn_pot(self, pred: jnp.ndarray, target: jnp.ndarray):
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
                sinkhorn2(a, b, loss_matrix, reg=0.1, numItermax=500, stopThr=1e-05),
                dtype=jnp.float32,
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
        """Euclidean distance matrix (pairwise)."""

        def dist(a, b):
            return jnp.sum(self._dist_fn(a, b) ** 2)

        if not squared:

            def dist(a, b):
                return jnp.sqrt(dist(a, b))

        return jnp.array(
            jax.vmap(lambda a: jax.vmap(lambda b: dist(a, b))(y))(x), dtype=jnp.float32
        )


def averaged_metrics(eval_metrics: MetricsDict) -> Dict[str, float]:
    """Averages the metrics over the rollouts."""
    # create a dictionary with the same keys as the metrics, but empty list as values
    trajectory_averages = defaultdict(list)
    for rollout in eval_metrics.values():
        for k, v in rollout.items():
            if k == "e_kin":
                v = v["mse"]
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
