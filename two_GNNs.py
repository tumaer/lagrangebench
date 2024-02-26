import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["JAX_ENABLE_X64"] = "True"
#export CUDA_VISIBLE_DEVICES=1

import lagrangebench
import haiku as hk
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import jax
import jax.numpy as jnp
from lagrangebench.data import H5Dataset
from lagrangebench.data.utils import numpy_collate
from lagrangebench.evaluate import averaged_metrics
from lagrangebench.defaults import defaults
from typing import Callable, Iterable, List, Optional, Tuple
from lagrangebench.evaluate.metrics import MetricsComputer, MetricsDict
from lagrangebench.utils import (
    broadcast_from_batch,
    get_kinematic_mask,
    load_haiku,
    set_seed,
)

rpf2d_test = lagrangebench.RPF2D("test", n_rollout_steps=20)

def pde_refiner(x):
    return lagrangebench.PDE_Refiner(
        problem_dimension=rpf2d_test.metadata["dim"],
        latent_size=128,
        number_of_layers=2,
        num_mp_steps=10,
        num_particle_types=9,  # 9 types (lagrangebench/utils.py)
        particle_type_embedding_size=16,  # 16 set to default
    )(x)
    

model = hk.without_apply_rng(hk.transform_with_state(pde_refiner))

#Case setup:
bounds = np.array(rpf2d_test.metadata["bounds"])
box = bounds[:, 1] - bounds[:, 0]

rpf_2d_case = lagrangebench.case_builder(
    box=box,  # (x,y) array with the world size along each axis. (1.0, 1.0) for 2D TGV
    metadata=rpf2d_test.metadata,  # metadata dictionary
    input_seq_length=6,  # number of consecutive time steps fed to the model
    isotropic_norm=False,  # whether to normalize each dimension independently
    noise_std=0.0,  # noise standard deviation used by the random-walk noise
    external_force_fn = rpf2d_test.external_force_fn
)

def two_models_infer(
    model: hk.TransformedWithState,
    case,
    data_test: H5Dataset,
    metrics: List = ["mse"],
    rollout_dir: Optional[str] = None,
    eval_n_trajs: int = defaults.eval_n_trajs,
    n_rollout_steps: int = defaults.n_rollout_steps,
    out_type: str = defaults.out_type,
    n_extrap_steps:  int = defaults.n_extrap_steps,
    seed: int = defaults.seed,
    metrics_stride: int = defaults.metrics_stride,
    **kwargs,):
    
    key, seed_worker, generator = set_seed(seed)
    
    loader_test = DataLoader(
        dataset=data_test,
        batch_size=1,
        collate_fn=numpy_collate,
        worker_init_fn=seed_worker,
        generator=generator,
        )
    
    metrics_computer = MetricsComputer(
        metrics,
        dist_fn=case.displacement,
        metadata=data_test.metadata,
        input_seq_length=data_test.input_seq_length,
        stride=metrics_stride,
    )
    
    model_apply = jax.jit(model.apply)
    
    # init values
    pos_input_and_target, particle_type = next(iter(loader_test))
    sample = (pos_input_and_target[0], particle_type[0])
    
    denoising_model_dir = os.path.join("ckp/rpf_2d_pde_ref_algo_1/pde_refiner_rpf2d_20240218-194222", "best")
    no_denoising_model_dir = os.path.join("ckp/rpf_2d_pde_ref_algo_1/pde_refiner_rpf2d_20240218-192702", "best")
    
    #denoising_model_dir = "ckp/rpf_2d_pde_ref_algo_1/pde_refiner_rpf2d_20240218-194222"
    #no_denoising_model_dir = "ckp/rpf_2d_pde_ref_algo_1/pde_refiner_rpf2d_20240218-192702"

    #Load the models
    params_denoising, state_denoising, _, _ = load_haiku(denoising_model_dir)
    params_base, state_base, _, _ = load_haiku(no_denoising_model_dir) 

    
    key, subkey = jax.random.split(key, 2)
    num_refinement_steps = kwargs["num_refinement_steps"]
    sigma_min = kwargs["sigma_min"]
    k  = jax.random.randint(subkey, (), 0, num_refinement_steps+1)
    is_k_zero = jnp.where(k==0, True, False)
    key, _, _, neighbors = case.allocate_pde_refiner(key, sample, k,is_k_zero,sigma_min,num_refinement_steps)
    
    #eval_rollout_pde_refiner() begins
    t_window = loader_test.dataset.input_seq_length #2
    eval_metrics = {}

    for i, traj_i in enumerate(loader_test): #for every trajectory in the test set
        # remove batch dimension
        assert traj_i[0].shape[0] == 1, "Batch dimension should be 1"
        traj_i = broadcast_from_batch(traj_i, index=0)  

        #eval_single_rollout_pde_refiner() begins
        
        pos_input, particle_type = traj_i #pos_input has the shape (3200,26,2) it loaded 20+6 positions for 3200 particles, 20 being the number of rollout steps and 6 is the input sequence length

        initial_positions = pos_input[:, 0:t_window]  # (n_nodes, t_window, dim), t_window = 2 
        traj_len = n_rollout_steps + 0  # (n_nodes, traj_len - t_window, dim)
        ground_truth_positions = pos_input[:, t_window : t_window + traj_len] #shape (3200,20,2)
        current_positions = initial_positions  
        n_nodes, _, dim = ground_truth_positions.shape 

        predictions = jnp.zeros((traj_len, n_nodes, dim)) 
        
        step = 0
        while step < n_rollout_steps + n_extrap_steps :  #runs 20 times
            sample = (current_positions, particle_type)
            features, neighbors = case.preprocess_eval_pde_refiner(sample, neighbors) #neighbour list is updated 
            
            if neighbors.did_buffer_overflow is True:
                edges_ = neighbors.idx.shape
                print(f"(eval) Reallocate neighbors list {edges_} at step {step}")
                _, neighbors = case.allocate_eval_pde_refiner(sample)  #if there is any overflow,then neighbour list is allocated
                print(f"(eval) To list {neighbors.idx.shape}")

                continue
            
            features['u_t_noised'] = jnp.zeros((features['vel_hist'].shape[0],2)) #0's
            features['k']= jnp.tile(0, (features['vel_hist'].shape[0],)) #set to 0
            
            #use the k=0 model for predicting the first value
            u_hat_t , _ = model_apply(params_base, state_base, (features, particle_type)) #predicts the 'acc' for gns and 'noise' for pde refiner

            max_refinement_steps=kwargs["num_refinement_steps"]
            min_noise_std = kwargs['sigma_min'] 
            #key = kwargs["key"]
            
            for k in range(1, max_refinement_steps+1): #Refinement loop
                
                key, subkey = jax.random.split(key, 2)

                noise_std =  min_noise_std**(k/max_refinement_steps)

                noise = jax.random.normal(subkey, jnp.zeros((features['vel_hist'].shape[0],2)).shape)

                features['u_t_noised'] = u_hat_t['noise'] + noise_std*noise

                #Modify the k value before sending it to the model
                features["k"] = jnp.tile(k, (features["vel_hist"].shape[0],))
                features["k"] = features["k"] * (1000 / max_refinement_steps)
                
                #use the denoising model for predicting the subsequent values
                pred, _ = model_apply(params_denoising, state_denoising, (features, particle_type))
                #pred is a dictionary with key 'noise' 
                u_hat_t['noise'] = features['u_t_noised'] - pred['noise']*noise_std

            refined_acc = {"acc": u_hat_t['noise']}

            next_position = case.integrate(refined_acc, current_positions)

            #Assuming n_extrap_steps = 0
            kinematic_mask = get_kinematic_mask(particle_type)
            next_position_ground_truth = ground_truth_positions[:, step]

            next_position = jnp.where(kinematic_mask[:, None], next_position_ground_truth, next_position)
            
            predictions = predictions.at[step].set(next_position) #shape of predictions: (20,3200,2)
            current_positions = jnp.concatenate([current_positions[:, 1:], next_position[:, None, :]], axis=1) #shape of current_positions: (3200,6,2)
            step += 1

        ground_truth_positions = ground_truth_positions.transpose(1, 0, 2)
        
        metrics = metrics_computer(predictions, ground_truth_positions)
        
        eval_metrics[f"rollout_{i}"] = metrics
        
        if (i + 1) == eval_n_trajs:
            break
    
    return eval_metrics

metrics = two_models_infer(  
    model,
    rpf_2d_case,
    rpf2d_test, 
    metrics=["mse"],  #mse, sink_horn and e_kin
    rollout_dir="rollouts/",
    eval_n_trajs=10, #Default is -1
    n_rollout_steps=20, 
    out_type="pkl", 
    n_extrap_steps=0,
    seed=0,
    metrics_stride=1, 
    num_refinement_steps = 3,
    sigma_min = 1e-7,
)

print(f"Averaged Metrics on Test split")
print(averaged_metrics(metrics))