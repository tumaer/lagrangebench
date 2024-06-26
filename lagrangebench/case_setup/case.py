"""Case setup functions."""

import warnings
from typing import Callable, Dict, Optional, Tuple, Union

import jax.numpy as jnp
from jax import Array, jit, lax, vmap
from jax_sph.jax_md import space
from jax_sph.jax_md.dataclasses import dataclass, static_field
from jax_sph.jax_md.partition import NeighborList, NeighborListFormat, neighbor_list
from omegaconf import DictConfig, OmegaConf

from lagrangebench.data.utils import get_dataset_stats
from lagrangebench.defaults import defaults
from lagrangebench.train.strats import add_gns_noise

from .features import FeatureDict, TargetDict, physical_feature_builder

TrainCaseOut = Tuple[Array, FeatureDict, TargetDict, NeighborList]
EvalCaseOut = Tuple[FeatureDict, NeighborList]
SampleIn = Tuple[jnp.ndarray, jnp.ndarray]

AllocateFn = Callable[[Array, SampleIn, float, int], TrainCaseOut]
AllocateEvalFn = Callable[[SampleIn], EvalCaseOut]

PreprocessFn = Callable[[Array, SampleIn, float, NeighborList, int], TrainCaseOut]
PreprocessEvalFn = Callable[[SampleIn, NeighborList], EvalCaseOut]

IntegrateFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


@dataclass
class CaseSetupFn:
    """Dataclass that contains all functions required to setup the case and simulate.

    Attributes:
        allocate: AllocateFn, runs the preprocessing without having a NeighborList as
            input.
        preprocess: PreprocessFn, takes positions from the dataloader, computes
            velocities, adds random-walk noise if needed, then updates the neighbor
            list, and return the inputs to the neural network as well as the targets.
        allocate_eval: AllocateEvalFn, same as allocate, but without noise addition
            and without targets.
        preprocess_eval: PreprocessEvalFn, same as allocate_eval, but jit-able.
        integrate: IntegrateFn, semi-implicit Euler integrations step respecting
            all boundary conditions.
        displacement: space.DisplacementFn, displacement function aware of boundary
            conditions (periodic on non-periodic).
        normalization_stats: Dict, normalization statisticss for input velocities and
            output acceleration.
    """

    allocate: AllocateFn = static_field()
    preprocess: PreprocessFn = static_field()
    allocate_eval: AllocateEvalFn = static_field()
    preprocess_eval: PreprocessEvalFn = static_field()
    integrate: IntegrateFn = static_field()
    displacement: space.DisplacementFn = static_field()
    normalization_stats: Dict = static_field()


def case_builder(
    box: Tuple[float, float, float],
    metadata: Dict,
    input_seq_length: int,
    cfg_neighbors: Union[Dict, DictConfig] = defaults.neighbors,
    cfg_model: Union[Dict, DictConfig] = defaults.model,
    noise_std: float = defaults.train.noise_std,
    external_force_fn: Optional[Callable] = None,
    dtype: jnp.dtype = defaults.dtype,
):
    """Set up a CaseSetupFn that contains every required function besides the model.

    Inspired by the `partition.neighbor_list` function in JAX-MD.

    The core functions are:
        * allocate, allocate memory for the neighbors list.
        * preprocess, update the neighbors list.
        * integrate, semi-implicit Euler respecting periodic boundary conditions.

    Args:
        box: Box xyz sizes of the system.
        metadata: Dataset metadata dictionary.
        input_seq_length: Length of the input sequence.
        cfg_neighbors: Configuration dictionary for the neighbor list.
        cfg_model: Configuration dictionary for the model / feature builder.
        noise_std: Noise standard deviation.
        external_force_fn: External force function.
        dtype: Data type.
    """
    if isinstance(cfg_neighbors, Dict):
        cfg_neighbors = OmegaConf.create(cfg_neighbors)
    if isinstance(cfg_model, Dict):
        cfg_model = OmegaConf.create(cfg_model)

    # if one of the cfg_* arguments has a subset of the default configs, merge them
    cfg_neighbors = OmegaConf.merge(defaults.neighbors, cfg_neighbors)
    cfg_model = OmegaConf.merge(defaults.model, cfg_model)

    normalization_stats = get_dataset_stats(
        metadata, cfg_model.isotropic_norm, noise_std
    )

    # apply PBC in all directions or not at all
    if jnp.array(metadata["periodic_boundary_conditions"]).any():
        displacement_fn, shift_fn = space.periodic(side=jnp.array(box))
    else:
        displacement_fn, shift_fn = space.free()

    displacement_fn_set = vmap(displacement_fn, in_axes=(0, 0))

    if cfg_neighbors.multiplier < 1.25:
        warnings.warn(
            f"cfg_neighbors.multiplier={cfg_neighbors.multiplier} < 1.25 is very low. "
            "Be especially cautious if you batch training and/or inference as "
            "reallocation might be necessary based on different overflow conditions. "
            "See https://github.com/tumaer/lagrangebench/pull/20#discussion_r1443811262"
        )

    neighbor_fn = neighbor_list(
        displacement_fn,
        jnp.array(box),
        backend=cfg_neighbors.backend,
        r_cutoff=metadata["default_connectivity_radius"],
        capacity_multiplier=cfg_neighbors.multiplier,
        mask_self=False,
        format=NeighborListFormat.Sparse,
        num_particles_max=metadata["num_particles_max"],
        pbc=metadata["periodic_boundary_conditions"],
    )

    feature_transform = physical_feature_builder(
        bounds=metadata["bounds"],
        normalization_stats=normalization_stats,
        connectivity_radius=metadata["default_connectivity_radius"],
        displacement_fn=displacement_fn,
        pbc=metadata["periodic_boundary_conditions"],
        magnitude_features=cfg_model.magnitude_features,
        external_force_fn=external_force_fn,
    )

    def _compute_target(pos_input: jnp.ndarray) -> TargetDict:
        # displacement(r1, r2) = r1-r2  # without PBC

        current_velocity = displacement_fn_set(pos_input[:, 1], pos_input[:, 0])
        next_velocity = displacement_fn_set(pos_input[:, 2], pos_input[:, 1])
        current_acceleration = next_velocity - current_velocity

        acc_stats = normalization_stats["acceleration"]
        normalized_acceleration = (
            current_acceleration - acc_stats["mean"]
        ) / acc_stats["std"]

        vel_stats = normalization_stats["velocity"]
        normalized_velocity = (next_velocity - vel_stats["mean"]) / vel_stats["std"]
        return {
            "acc": normalized_acceleration,
            "vel": normalized_velocity,
            "pos": pos_input[:, -1],
        }

    def _preprocess(
        sample: Tuple[jnp.ndarray, jnp.ndarray],
        neighbors: Optional[NeighborList] = None,
        is_allocate: bool = False,
        mode: str = "train",
        **kwargs,  # key, noise_std, unroll_steps
    ) -> Union[TrainCaseOut, EvalCaseOut]:
        pos_input = jnp.asarray(sample[0], dtype=dtype)
        particle_type = jnp.asarray(sample[1])

        if mode == "train":
            key, noise_std = kwargs["key"], kwargs["noise_std"]
            unroll_steps = kwargs["unroll_steps"]
            if pos_input.shape[1] > 1:
                key, pos_input = add_gns_noise(
                    key, pos_input, particle_type, input_seq_length, noise_std, shift_fn
                )

        # allocate the neighbor list
        most_recent_position = pos_input[:, input_seq_length - 1]
        num_particles = (particle_type != -1).sum()
        if is_allocate:
            neighbors = neighbor_fn.allocate(
                most_recent_position, num_particles=num_particles
            )
        else:
            neighbors = neighbors.update(
                most_recent_position, num_particles=num_particles
            )

        # selected features
        features = feature_transform(pos_input[:, :input_seq_length], neighbors)

        if mode == "train":
            # compute target acceleration. Inverse of postprocessing step.
            # the "-2" is needed because we need the most recent position and one before
            slice_begin = (0, input_seq_length - 2 + unroll_steps, 0)
            slice_size = (pos_input.shape[0], 3, pos_input.shape[2])

            target_dict = _compute_target(
                lax.dynamic_slice(pos_input, slice_begin, slice_size)
            )
            return key, features, target_dict, neighbors
        if mode == "eval":
            return features, neighbors

    def allocate_fn(key, sample, noise_std=0.0, unroll_steps=0):
        return _preprocess(
            sample,
            key=key,
            noise_std=noise_std,
            unroll_steps=unroll_steps,
            is_allocate=True,
        )

    @jit
    def preprocess_fn(key, sample, noise_std, neighbors, unroll_steps=0):
        return _preprocess(
            sample, neighbors, key=key, noise_std=noise_std, unroll_steps=unroll_steps
        )

    def allocate_eval_fn(sample):
        return _preprocess(sample, is_allocate=True, mode="eval")

    @jit
    def preprocess_eval_fn(sample, neighbors):
        return _preprocess(sample, neighbors, mode="eval")

    @jit
    def integrate_fn(normalized_in, position_sequence):
        """Euler integrator to get position shift."""
        assert any([key in normalized_in for key in ["pos", "vel", "acc"]])

        if "pos" in normalized_in:
            # Zeroth euler step
            return normalized_in["pos"]
        else:
            most_recent_position = position_sequence[:, -1]
            if "vel" in normalized_in:
                # invert normalization
                velocity_stats = normalization_stats["velocity"]
                new_velocity = velocity_stats["mean"] + (
                    normalized_in["vel"] * velocity_stats["std"]
                )
            elif "acc" in normalized_in:
                # invert normalization.
                acceleration_stats = normalization_stats["acceleration"]
                acceleration = acceleration_stats["mean"] + (
                    normalized_in["acc"] * acceleration_stats["std"]
                )
                # Second Euler step
                most_recent_velocity = displacement_fn_set(
                    most_recent_position, position_sequence[:, -2]
                )
                new_velocity = most_recent_velocity + acceleration  # * dt = 1

            # First Euler step
            return shift_fn(most_recent_position, new_velocity)

    return CaseSetupFn(
        allocate_fn,
        preprocess_fn,
        allocate_eval_fn,
        preprocess_eval_fn,
        integrate_fn,
        displacement_fn,
        normalization_stats,
    )
