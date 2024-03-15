"""Case setup functions."""

import warnings
from functools import partial
from typing import Callable, Dict, Optional, Tuple, Union

import jax.numpy as jnp
from jax import Array, jit, lax, random, vmap
from jax_md import space
from jax_md.dataclasses import dataclass, static_field
from jax_md.partition import NeighborList, NeighborListFormat

from lagrangebench.data.utils import get_dataset_stats
from lagrangebench.defaults import defaults
from lagrangebench.train.strats import add_gns_noise
from lagrangebench.utils import ACDMConfig

from .features import FeatureDict, TargetDict, physical_feature_builder
from .partition import neighbor_list

TrainCaseOut = Tuple[Array, FeatureDict, TargetDict, NeighborList]
EvalCaseOut = Tuple[FeatureDict, NeighborList]
SampleIn = Tuple[jnp.ndarray, jnp.ndarray]

AllocateFn = Callable[[Array, SampleIn, float, int], TrainCaseOut]
AllocateEvalFn = Callable[[SampleIn], EvalCaseOut]

PreprocessFn = Callable[[Array, SampleIn, float, NeighborList, int], TrainCaseOut]
PreprocessEvalFn = Callable[[SampleIn, NeighborList], EvalCaseOut]

# For PDE Refiner
AllocatePdeRefinerFn = Callable[[Array, SampleIn, int, bool, float, int, float, int], TrainCaseOut]
AllocateEvalPdeRefinerFn = Callable[[SampleIn], EvalCaseOut]
PreprocessPdeRefinerFn = Callable[
    [Array, SampleIn, float, NeighborList, int, bool, float, int, int], TrainCaseOut
]
PreprocessEvalPdeRefinerFn = Callable[[SampleIn, NeighborList], EvalCaseOut]

#For ACDM
AllocateAcdmFn = Callable[[Array, SampleIn, int, ACDMConfig, float, int], TrainCaseOut]
AllocateEvalAcdmFn = Callable[[SampleIn], EvalCaseOut]
PreprocessAcdmFn = Callable[
    [random.KeyArray, SampleIn, float, NeighborList, int], TrainCaseOut
]
PreprocessEvalAcdmFn = Callable[[SampleIn, NeighborList], EvalCaseOut]

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
    allocate_pde_refiner: AllocatePdeRefinerFn = static_field()
    preprocess_pde_refiner: PreprocessPdeRefinerFn = static_field()
    allocate_eval_pde_refiner: AllocateEvalPdeRefinerFn = static_field()
    preprocess_eval_pde_refiner: PreprocessEvalPdeRefinerFn = static_field()
    allocate_acdm: AllocateAcdmFn = static_field()
    preprocess_acdm: PreprocessAcdmFn = static_field()
    allocate_eval_acdm: AllocateEvalAcdmFn = static_field()
    preprocess_eval_acdm: PreprocessEvalAcdmFn = static_field()
    integrate: IntegrateFn = static_field()
    displacement: space.DisplacementFn = static_field()
    normalization_stats: Dict = static_field()


def case_builder(
    box: Tuple[float, float, float],
    metadata: Dict,
    input_seq_length: int,
    isotropic_norm: bool = defaults.isotropic_norm,
    noise_std: float = defaults.noise_std,
    external_force_fn: Optional[Callable] = None,
    magnitude_features: bool = defaults.magnitude_features,
    neighbor_list_backend: str = defaults.neighbor_list_backend,
    neighbor_list_multiplier: float = defaults.neighbor_list_multiplier,
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
        isotropic_norm: Whether to normalize dimensions equally.
        noise_std: Noise standard deviation.
        external_force_fn: External force function.
        magnitude_features: Whether to add velocity magnitudes in the features.
        neighbor_list_backend: Backend of the neighbor list.
        neighbor_list_multiplier: Capacity multiplier of the neighbor list.
        dtype: Data type.
    """
    normalization_stats = get_dataset_stats(metadata, isotropic_norm, noise_std)

    # apply PBC in all directions or not at all
    if jnp.array(metadata["periodic_boundary_conditions"]).any():
        displacement_fn, shift_fn = space.periodic(side=jnp.array(box))
    else:
        displacement_fn, shift_fn = space.free()

    displacement_fn_set = vmap(displacement_fn, in_axes=(0, 0))

    if neighbor_list_multiplier < 1.25:
        warnings.warn(
            f"neighbor_list_multiplier={neighbor_list_multiplier} < 1.25 is very low. "
            "Be especially cautious if you batch training and/or inference as "
            "reallocation might be necessary based on different overflow conditions. "
            "See https://github.com/tumaer/lagrangebench/pull/20#discussion_r1443811262"
        )

    neighbor_fn = neighbor_list(
        displacement_fn,
        jnp.array(box),
        backend=neighbor_list_backend,
        r_cutoff=metadata["default_connectivity_radius"],
        capacity_multiplier=neighbor_list_multiplier,
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
        magnitude_features=magnitude_features,
        external_force_fn=external_force_fn,
    )

    def _compute_target(pos_input: jnp.ndarray) -> TargetDict:
        # displacement(r1, r2) = r1-r2  # without PBC
        # returns the target dictionary with the target position, velocity and acc.
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

    # Additional functions for PDE Refiner
    def _preprocess_pde_refiner(
        sample: Tuple[jnp.ndarray, jnp.ndarray],
        neighbors: Optional[NeighborList] = None,
        is_allocate: bool = False,  # bool to allocate new Neighbour List
        mode: str = "train",
        **kwargs,
    ) -> Union[TrainCaseOut, EvalCaseOut]:
        pos_input = jnp.asarray(
            sample[0], dtype=dtype
        )  # shape of sample[0] == pos_input = (3200,7,2).
        particle_type = jnp.asarray(sample[1])

        if mode == "train":
            key, noise_std = kwargs["key"], kwargs["noise_std"]
            unroll_steps = kwargs["unroll_steps"]
            if pos_input.shape[1] > 1:  # pos_input.shape[1] = 7
                key, pos_input = add_gns_noise(
                    key, pos_input, particle_type, input_seq_length, noise_std, shift_fn
                )

        # allocate the neighbor list
        most_recent_position = pos_input[
            :, input_seq_length - 1
        ]  # input_seq_length = 6
        num_particles = (particle_type != -1).sum()
        if is_allocate:
            neighbors = neighbor_fn.allocate(
                most_recent_position, num_particles=num_particles
            )
        else:
            neighbors = neighbors.update(
                most_recent_position, num_particles=num_particles
            )

        features = feature_transform(pos_input[:, :input_seq_length], neighbors)

        if mode == "train":
            key = kwargs["key"]
            k = kwargs["k"]
            is_k_zero = kwargs["is_k_zero"]
            
            #can be 'acc' or 'vel'
            refinement_parameter = kwargs["refinement_parameter"]
            
            key, subkey = random.split(key, 2)

            min_noise_std = kwargs["sigma_min"]
            max_refinement_steps = kwargs["num_refinement_steps"]

            features["k"] = jnp.tile(k, (features["vel_hist"].shape[0],))

            slice_begin = (
                0,
                input_seq_length - 2 + unroll_steps,
                0,
            )
            # =(0,0,0) if unroll_steps = 0 and input_seq_length = 2
            slice_size = (pos_input.shape[0], 3, pos_input.shape[2])

            # target_dict has the target position, velocity and acceleration
            target_dict = _compute_target(
                lax.dynamic_slice(pos_input, slice_begin, slice_size)
            )
            if is_k_zero:
                features["noised_data"] = jnp.zeros((features["vel_hist"].shape[0], 2))
                target_dict["noise"] = target_dict[refinement_parameter]

            else:
                noise_std = min_noise_std ** (k / max_refinement_steps)

                noise = random.normal(
                    subkey, jnp.zeros((features["vel_hist"].shape[0], 2)).shape
                )
                features["noised_data"] = target_dict[refinement_parameter] + noise_std * noise
                target_dict["noise"] = noise

            #
            return (
                key,
                features,
                target_dict,
                neighbors,
            )

        if mode == "eval":
            return features, neighbors

    def allocate_pde_refiner_fn(
        key,
        sample,
        k,
        is_k_zero,
        sigma_min,
        num_refinement_steps,
        refinement_parameter,
        noise_std=0.0,
        unroll_steps=0,
    ):
        return _preprocess_pde_refiner(
            sample,
            key=key,
            k=k,
            is_k_zero=is_k_zero,
            sigma_min=sigma_min,
            num_refinement_steps=num_refinement_steps,
            noise_std=noise_std,
            refinement_parameter=refinement_parameter,
            unroll_steps=unroll_steps,
            is_allocate=True,
        )

    @partial(jit, static_argnames=["is_k_zero", "num_refinement_steps", "refinement_parameter"])
    def preprocess_pde_refiner_fn(
        key,
        sample,
        noise_std,
        neighbors,
        k,
        is_k_zero,
        sigma_min,
        num_refinement_steps,
        refinement_parameter,
        unroll_steps=0,
    ):
        return _preprocess_pde_refiner(
            sample,
            neighbors,
            key=key,
            k=k,
            is_k_zero=is_k_zero,
            sigma_min=sigma_min,
            num_refinement_steps=num_refinement_steps,
            noise_std=noise_std,
            refinement_parameter=refinement_parameter,
            unroll_steps=unroll_steps,
        )

    def allocate_eval_pde_refiner_fn(sample):
        return _preprocess_pde_refiner(sample, is_allocate=True, mode="eval")

    @jit
    def preprocess_eval_pde_refiner_fn(sample, neighbors):
        return _preprocess_pde_refiner(sample, neighbors, mode="eval")

    # Additional functions for ACDM
    def compute_acc_based_on_pos_slice(pos_input: jnp.ndarray):
        """Compute acceleration based on position slice."""
        current_velocity = displacement_fn_set(pos_input[:, 1], pos_input[:, 0])
        next_velocity = displacement_fn_set(pos_input[:, 2], pos_input[:, 1])
        current_acceleration = next_velocity - current_velocity
        acc_stats = normalization_stats["acceleration"]
        return (current_acceleration - acc_stats["mean"]) / acc_stats["std"]

    def compute_vel_based_on_pos_slice(pos_input: jnp.ndarray):
        """Compute velocity based on position slice."""
        current_velocity = displacement_fn_set(pos_input[:, 1], pos_input[:, 0])
        vel_stats = normalization_stats["velocity"]
        return (current_velocity - vel_stats["mean"]) / vel_stats["std"]

    def extract_conditioning_data(pos_input: jnp.ndarray, 
                                  num_conditioning_steps: int, 
                                  conditioning_parameter: str):
        """Extract conditioning data for ACDM."""
        conditioning_data={}
        if conditioning_parameter == "acc":
            assert input_seq_length >= num_conditioning_steps + 2
        elif conditioning_parameter == "vel":
            assert input_seq_length >= num_conditioning_steps + 1
        
        for i in range(num_conditioning_steps):
            
            if conditioning_parameter == "acc":
                slice_begin = (0, input_seq_length - num_conditioning_steps -2 + i, 0) #(0,2,0)
                slice_size = (pos_input.shape[0], 3, pos_input.shape[2]) #(3200,3,2)
                value = compute_acc_based_on_pos_slice(
                    lax.dynamic_slice(pos_input, slice_begin, slice_size)
                )
                key = f"acc_t_minus_{num_conditioning_steps - i}"
                conditioning_data[key] = value
                
            elif conditioning_parameter == "vel":
                slice_begin = (0, input_seq_length - num_conditioning_steps -1 + i, 0)
                slice_size = (pos_input.shape[0], 2, pos_input.shape[2]) 
                value = compute_vel_based_on_pos_slice(
                    lax.dynamic_slice(pos_input, slice_begin, slice_size)
                )
                key = f"vel_t_minus_{num_conditioning_steps - i}"
                conditioning_data[key] = value

        return conditioning_data 

    def _preprocess_acdm(
        sample: Tuple[jnp.ndarray, jnp.ndarray],
        neighbors: Optional[NeighborList] = None,
        is_allocate: bool = False,
        mode: str = "train",
        **kwargs,
    ) -> Union[TrainCaseOut, EvalCaseOut]:
        pos_input = jnp.asarray(
            sample[0], dtype=dtype
        )  # shape of sample[0] == pos_input = (3200,7,2) iff input_seq_length = 6.
        particle_type = jnp.asarray(sample[1])

        if mode == "train":
            key, noise_std = kwargs["key"], kwargs["noise_std"]
            unroll_steps = kwargs["unroll_steps"]
            if pos_input.shape[1] > 1:  # pos_input.shape[1] = 7
                key, pos_input = add_gns_noise(
                    key, pos_input, particle_type, input_seq_length, noise_std, shift_fn
                )

        # allocate the neighbor list
        most_recent_position = pos_input[
            :, input_seq_length - 1
        ]  # input_seq_length = 6
        num_particles = (particle_type != -1).sum()
        if is_allocate:
            neighbors = neighbor_fn.allocate(
                most_recent_position, num_particles=num_particles
            )
        else:
            neighbors = neighbors.update(
                most_recent_position, num_particles=num_particles
            )

        features = feature_transform(pos_input[:, :input_seq_length], neighbors)

        if mode == "train":
            key = kwargs["key"]
            acdm_config = kwargs["acdm_config"]
            
            num_conditioning_steps = acdm_config.num_conditioning_steps
            conditioning_parameter = acdm_config.conditioning_parameter
            
            key, subkey = random.split(key, 2)

            slice_begin = (
                0,
                input_seq_length - 2 + unroll_steps,
                0,
            )
            # =(0,0,0) if unroll_steps = 0 and input_seq_length = 2
            slice_size = (pos_input.shape[0], 3, pos_input.shape[2])

            # target_dict has the target position, velocity and acceleration
            target_dict = _compute_target(
                lax.dynamic_slice(pos_input, slice_begin, slice_size)
            )

            # Compute two previous acceleration and concatenate with the target
            # acceleration. All values are normalized.
            conditioning_data = extract_conditioning_data(pos_input, 
                                                      num_conditioning_steps,
                                                      conditioning_parameter)

            conditioning_data = jnp.concatenate(list(conditioning_data.values()), axis=1)
            #dictionary containing the conditioning data and the target which is required for training
            if conditioning_parameter == "acc":
                features["concatenated_data"] = jnp.concatenate(
                    (
                        conditioning_data,
                        target_dict["acc"],
                    ),
                    axis=1,
                )
            else:    
                features["concatenated_data"] = jnp.concatenate(
                    (
                        conditioning_data,
                        target_dict["vel"],
                    ),
                    axis=1,
                )

            # Sample noise from a normal distribution
            noise = random.normal(
                subkey,
                jnp.zeros(
                    (features["concatenated_data"].shape[0], features["concatenated_data"].shape[1])
                ).shape,
            )
            target_dict["noise"] = noise
            # Sample a random timestep between 0 and number of diffusion steps
            
            features["k"] = jnp.tile(kwargs["k"], (features["vel_hist"].shape[0],))
            # Perform Forward Diffusion step to obtain the dNoisy
            features["noised_data"] = acdm_config.sqrtAlphasCumprod[kwargs["k"]] * features["concatenated_data"] \
            + acdm_config.sqrtOneMinusAlphasCumprod[kwargs["k"]] * noise

            return (
                key,
                features,
                target_dict,
                neighbors,
            )

        if mode == "eval":
            return features, neighbors

    def allocate_acdm_fn(
        key,
        sample,
        k,
        acdm_config,
        noise_std=0.0,
        unroll_steps=0,
    ):
        return _preprocess_acdm(
            sample,
            key=key,
            k=k,
            acdm_config=acdm_config,
            noise_std=noise_std,
            unroll_steps=unroll_steps,
            is_allocate=True,
        )
    @partial(jit, static_argnames=["acdm_config"])
    def preprocess_acdm_fn(        
        key,
        sample,
        noise_std,
        neighbors,
        k,
        acdm_config,
        unroll_steps=0,):
        return _preprocess_acdm(
            sample,
            neighbors,
            key=key,
            k=k,
            acdm_config=acdm_config,
            noise_std=noise_std,
            unroll_steps=unroll_steps,
        )

    def allocate_eval_acdm_fn(sample):
        return _preprocess_pde_refiner(sample, is_allocate=True, mode="eval")

    @jit
    def preprocess_eval_acdm_fn(sample, neighbors):
        return _preprocess_pde_refiner(sample, neighbors, mode="eval")

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
        allocate_pde_refiner_fn,
        preprocess_pde_refiner_fn,
        allocate_eval_pde_refiner_fn,
        preprocess_eval_pde_refiner_fn,
        allocate_acdm_fn,
        preprocess_acdm_fn,
        allocate_eval_acdm_fn,
        preprocess_eval_acdm_fn,
        integrate_fn,
        displacement_fn,
        normalization_stats,
    )
