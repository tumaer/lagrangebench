"""Case setup functions."""

from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import jax.numpy as jnp
from jax import jit, lax, random, vmap
from jax_md import space
from jax_md.dataclasses import dataclass, static_field
from jax_md.partition import NeighborList, NeighborListFormat
from functools import partial
import time

from lagrangebench.data.utils import get_dataset_stats
from lagrangebench.defaults import defaults
from lagrangebench.train.strats import add_gns_noise

from .features import FeatureDict, TargetDict, physical_feature_builder
from .partition import neighbor_list

TrainCaseOut = Tuple[random.KeyArray, FeatureDict, TargetDict, NeighborList]
EvalCaseOut = Tuple[FeatureDict, NeighborList]
SampleIn = Tuple[jnp.ndarray, jnp.ndarray]

AllocateFn = Callable[[random.KeyArray, SampleIn, float, int], TrainCaseOut]
AllocateEvalFn = Callable[[SampleIn], EvalCaseOut]

PreprocessFn = Callable[[random.KeyArray, SampleIn, float, NeighborList, int], TrainCaseOut]
PreprocessEvalFn = Callable[[SampleIn, NeighborList], EvalCaseOut]

#For PDE Refiner
AllocatePdeRefinerFn = Callable[[random.KeyArray, SampleIn, float, int], TrainCaseOut]
AllocateEvalPdeRefinerFn = Callable[[SampleIn], EvalCaseOut]
PreprocessPdeRefinerFn = Callable[[random.KeyArray, SampleIn, float, NeighborList, int], TrainCaseOut]
PreprocessEvalPdeRefinerFn = Callable[[SampleIn, NeighborList], EvalCaseOut]

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
    integrate: IntegrateFn = static_field()
    displacement: space.DisplacementFn = static_field()
    normalization_stats: Dict = static_field()


def case_builder(
    box: Tuple[float, float, float],
    metadata: Dict, # extracted from metadata.json (datasets/2DRPF_3200_20kevery100/metadata.json)
    input_seq_length: int,  #default= 6 position values
    isotropic_norm: bool = defaults.isotropic_norm,  #False, set in lagrangebench/defaults.py (For RPF_2D_3200_20kevery100)
    noise_std: float = defaults.noise_std,  #default = 3e-4, set in lagrangebench/defaults.py (For RPF_2D_3200_20kevery100)
    external_force_fn: Optional[Callable] = None, #True for RPF_2D_3200_20kevery100. Mentioned in experiments/utils.py
    magnitude_features: bool = defaults.magnitude_features, #Set to False for RPF_2D_3200_20kevery100
    neighbor_list_backend: str = defaults.neighbor_list_backend, # Options: jaxmd_vmap, jaxmd_scan, matscipy. Jaxmd_vmap is the default
    neighbor_list_multiplier: float = defaults.neighbor_list_multiplier, #set to 1.25 in defaults.yaml (For RPF_2D_3200_20kevery100)
    dtype: jnp.dtype = defaults.dtype, #f64 or f32 depending on the defaults.yaml
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
        isotropic_norm: Whether to use isotropic normalization.
        noise_std: Noise standard deviation.
        external_force_fn: External force function.
        magnitude_features: Whether to add velocity magnitudes in the features.
        neighbor_list_backend: Backend of the neighbor list.
        neighbor_list_multiplier: Capacity multiplier of the neighbor list.
        dtype: Data type.
    """
    normalization_stats = get_dataset_stats(metadata, isotropic_norm, noise_std) #copied from metadata.json (datasets/2DRPF_3200_20kevery100/metadata.json)

    # apply PBC in all directions or not at all
    if jnp.array(metadata["periodic_boundary_conditions"]).any():
        displacement_fn, shift_fn = space.periodic(side=jnp.array(box))
    else:
        displacement_fn, shift_fn = space.free()

    displacement_fn_set = vmap(displacement_fn, in_axes=(0, 0))

    neighbor_fn = neighbor_list(
        displacement_fn,
        jnp.array(box),
        backend=neighbor_list_backend,
        r_cutoff=metadata["default_connectivity_radius"], #default_connectivity_radius=0.036
        capacity_multiplier=neighbor_list_multiplier, #=1.25 default
        mask_self=False,
        format=NeighborListFormat.Sparse,
        num_particles_max=metadata["num_particles_max"],
        pbc=metadata["periodic_boundary_conditions"],
    )
    #refer to lagrangebench/case_setup/features.py
    feature_transform = physical_feature_builder(
        bounds=metadata["bounds"],
        normalization_stats=normalization_stats,
        connectivity_radius=metadata["default_connectivity_radius"],
        displacement_fn=displacement_fn,
        pbc=metadata["periodic_boundary_conditions"],
        magnitude_features=magnitude_features, #set to False for the dataset: RPF_2D_3200_20kevery100
        external_force_fn=external_force_fn,
    )

    def _compute_target(pos_input: jnp.ndarray) -> TargetDict:  #Computes the velocity and acceleration from position data
        # here we have the last three positions as pos_input: (3200,3,2) from (3200,7,2)<--original pos_input
        # Need for 3 time step data of position as we need to compute current velocity and next velocity to get current acceleration
        # displacement(r1, r2) = r1-r2  # without PBC
        
        current_velocity = displacement_fn_set(pos_input[:, 1], pos_input[:, 0])
        next_velocity = displacement_fn_set(pos_input[:, 2], pos_input[:, 1])
        current_acceleration = next_velocity - current_velocity

        acc_stats = normalization_stats["acceleration"] #containis mean and std. of acceleration
        normalized_acceleration = (current_acceleration - acc_stats["mean"]) / acc_stats["std"]

        vel_stats = normalization_stats["velocity"]
        normalized_velocity = (next_velocity - vel_stats["mean"]) / vel_stats["std"]
        #returns the target variables which will serve as ground truth
        return {
            "acc": normalized_acceleration, # Acceleration at t+1 where t=6
            "vel": normalized_velocity,  # Velocity at t=7
            "pos": pos_input[:, -1], # Position at t=7
        }

    def _preprocess(
        sample: Tuple[jnp.ndarray, jnp.ndarray],
        neighbors: Optional[NeighborList] = None,
        is_allocate: bool = False, #bool to allocate new Neighbour List
        mode: str = "train",  #can be either train or eval
        **kwargs,  # key, noise_std
    ) -> Union[TrainCaseOut, EvalCaseOut]:
        pos_input = jnp.asarray(sample[0], dtype=dtype) #shape of sample[0] == pos_input = (3200,7,2). 
                                                        # 7 time step data is provided, because we need the target (ground truth) position as well
        particle_type = jnp.asarray(sample[1])  #shape of sample[1] == particle_type = (3200,)

        #For every particle out of 3200, we have 7 historic positions, each with x and y coordinates. 
        if mode == "train":
            key, noise_std = kwargs["key"], kwargs["noise_std"]
            unroll_steps = kwargs["unroll_steps"]
            if pos_input.shape[1] > 1:  #pos_input.shape[1] = 7
                #random_seed = np.random.randint(0, 2**32 - 1)  # Use a 32-bit integer as the seed
                # Initialize a PRNG key using the random seed
                #key2 = random.PRNGKey(random_seed)
                #noise_std = random.choice(key2,jnp.array([0.001,0.0003,0.0001]))
                key, pos_input = add_gns_noise(
                    key, pos_input, particle_type, input_seq_length, noise_std, shift_fn
                )

        # allocate the neighbor list
        most_recent_position = pos_input[:, input_seq_length - 1]  #input_seq_length = 6. '-1' is because the index starts from 0
        num_particles = (particle_type != -1).sum()
        if is_allocate:
            neighbors = neighbor_fn.allocate(most_recent_position, num_particles=num_particles)
        else:
            neighbors = neighbors.update(most_recent_position, num_particles=num_particles)

        # selected features, for RPF_2D_3200_20kevery100, feature_transform returns a dictionary with keys: 
        # ['abs_pos', 'force', 'receivers', 'rel_disp', 'rel_dist', 'senders', 'vel_hist']
        features = feature_transform(pos_input[:, :input_seq_length], neighbors)


        if mode == "train":
            # compute target acceleration. Inverse of postprocessing step.
            # the "-2" is needed because we need the most recent position and one before
            slice_begin = (0, input_seq_length - 2 + unroll_steps, 0)  #=(0,0,0) if unroll_steps = 0 and input_seq_length = 2
            slice_size = (pos_input.shape[0], 3, pos_input.shape[2])   #=(3200,3,2) 
                                                                      
            #target_dict has the target position, velocity and acceleration i.e. we can extract u(t) from this dictionary
            target_dict = _compute_target(lax.dynamic_slice(pos_input, slice_begin, slice_size))
            #output of lax.dynamic_slice(pos_input, slice_begin, slice_size) = (3200,3,2), the last three timestep data
            
            return key, features, target_dict, neighbors #target_dict returned only for training and not for CV or testing
        
        if mode == "eval":
            return features, neighbors

    # For 'allocate_fn' and 'allocate_eval_fn', neighbour_list is to be generated, so it is not passed as an argument
    def allocate_fn(key, sample, noise_std=0.0, unroll_steps=0): #while initializing the case, the noise_std is 0.0, but inside the training loop in trainer.py, it is set properly.
        return _preprocess(
            sample,
            key=key,
            noise_std=noise_std,
            unroll_steps=unroll_steps,
            is_allocate=True,
        )

    @jit  #For jitted 'preprocess_fn' and 'preprocess_eval_fn', neighbour_list is not updated and passed as argument.
    def preprocess_fn(key, sample, noise_std, neighbors, unroll_steps=0):
        return _preprocess(
            sample, neighbors, key=key, noise_std=noise_std, unroll_steps=unroll_steps
        )

    def allocate_eval_fn(sample): #new neighbour list is to be generated and there is no noise addition
        return _preprocess(sample, is_allocate=True, mode="eval")

    @jit
    def preprocess_eval_fn(sample, neighbors):# no new neighbourlist and no noise.
        return _preprocess(sample, neighbors, mode="eval")


    #Additional functions for PDE Refiner
    def _preprocess_pde_refiner(sample: Tuple[jnp.ndarray, jnp.ndarray],
        neighbors: Optional[NeighborList] = None,
        is_allocate: bool = False, #bool to allocate new Neighbour List
        mode: str = "train", 
        **kwargs, 
    ) -> Union[TrainCaseOut, EvalCaseOut]:
        pos_input = jnp.asarray(sample[0], dtype=dtype) #shape of sample[0] == pos_input = (3200,3,2). 
        particle_type = jnp.asarray(sample[1])
        
        if mode == "train":
            key, noise_std = kwargs["key"], kwargs["noise_std"]
            unroll_steps = kwargs["unroll_steps"]
            if pos_input.shape[1] > 1:  #pos_input.shape[1] = 7    
                key, pos_input = add_gns_noise(key, pos_input, particle_type, input_seq_length, noise_std, shift_fn)
        
        # allocate the neighbor list
        most_recent_position = pos_input[:, input_seq_length - 1]  #input_seq_length = 6
        num_particles = (particle_type != -1).sum()
        if is_allocate:
            neighbors = neighbor_fn.allocate(most_recent_position, num_particles=num_particles)
        else:
            neighbors = neighbors.update(most_recent_position, num_particles=num_particles)

        features = feature_transform(pos_input[:, :input_seq_length], neighbors) #has only u(t-dt)
        
        if mode == "train": 
            key = kwargs["key"]
            k = kwargs["k"]
            is_k_zero = kwargs["is_k_zero"]
            
            key, subkey = random.split(key, 2)
            
            min_noise_std = kwargs['sigma_min'] 
            max_refinement_steps=kwargs["num_refinement_steps"]
            
            features['k'] = jnp.tile(k, (features['vel_hist'].shape[0],)) #shape = (3200,1)
            
            if(max_refinement_steps!=0):
                features['k'] = features['k']*(1000/max_refinement_steps)
            
            
            slice_begin = (0, input_seq_length - 2 + unroll_steps, 0)  #=(0,0,0) if unroll_steps = 0 and input_seq_length = 2
            slice_size = (pos_input.shape[0], 3, pos_input.shape[2])   #=(3200,3,2) 

            #target_dict has the target position, velocity and acceleration i.e. we can extract u(t) from this dictionary
            target_dict = _compute_target(lax.dynamic_slice(pos_input, slice_begin, slice_size))

            if is_k_zero:
                features['u_t_noised'] = jnp.zeros((features['vel_hist'].shape[0],2))
                target_dict['noise'] = target_dict['acc']

            else:
                noise_std = min_noise_std**(k/max_refinement_steps)
                noise = random.normal(subkey, jnp.zeros((features['vel_hist'].shape[0],2)).shape)   #sampled from gaussian distribution
                features['u_t_noised'] = target_dict['acc'] + noise_std*noise
                target_dict['noise'] = noise 
            
            #output of lax.dynamic_slice(pos_input, slice_begin, slice_size) = (3200,3,2), the last three timestep data
            return key, features, target_dict, neighbors #target_dict returned only for training and not for CV or testing
        
        if mode == "eval": 
            return features, neighbors

    def allocate_pde_refiner_fn(key,sample,k,is_k_zero,sigma_min,num_refinement_steps,noise_std=0.0, unroll_steps=0): #while initializing the case, the noise_std is 0.0, but inside the training loop in trainer.py, it is set properly.
        return _preprocess_pde_refiner(sample,key=key,k=k, is_k_zero=is_k_zero,sigma_min=sigma_min, num_refinement_steps=num_refinement_steps, noise_std=noise_std, unroll_steps=unroll_steps, is_allocate=True,
        )

    @partial(jit, static_argnames=['is_k_zero', 'num_refinement_steps'])  #For jitted 'preprocess_fn' and 'preprocess_eval_fn', neighbour_list is not updated and passed as argument.
    def preprocess_pde_refiner_fn(key, sample, noise_std, neighbors,k, is_k_zero,sigma_min, num_refinement_steps,unroll_steps=0):
        return _preprocess_pde_refiner(
            sample, neighbors, key=key,  k=k, is_k_zero=is_k_zero, sigma_min=sigma_min, num_refinement_steps=num_refinement_steps, noise_std=noise_std, unroll_steps=unroll_steps,
        )

    def allocate_eval_pde_refiner_fn(sample): #new neighbour list is to be generated and there is no noise addition
        return _preprocess_pde_refiner(sample, is_allocate=True, mode="eval")

    @jit
    def preprocess_eval_pde_refiner_fn(sample, neighbors):# no new neighbourlist and no noise.
        return _preprocess_pde_refiner(sample, neighbors, mode="eval")
    

    @jit
    def integrate_fn(normalized_in, position_sequence): #Semi Implicit Euler Integrator used in rollout.py as case.integrate
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
        integrate_fn,
        displacement_fn,
        normalization_stats,
    )
