import copy
import os
import os.path as osp
from argparse import Namespace
from datetime import datetime

import haiku as hk
import jax.numpy as jnp
import jmp
import numpy as np
import yaml

import wandb
from experiments.utils import setup_data, setup_model
from lagrangebench import Trainer, infer, infer_pde_refiner
from lagrangebench.case_setup import case_builder
from lagrangebench.evaluate import averaged_metrics
from lagrangebench.utils import PushforwardConfig

#invoked directly from main
def train_or_infer(args: Namespace):
    data_train, data_eval, args = setup_data(args)
    
    # neighbors search
    bounds = np.array(data_train.metadata["bounds"])
    args.box = bounds[:, 1] - bounds[:, 0]

    args.info.len_train = len(data_train)  #for RPF 2d: 19995
    args.info.len_eval = len(data_eval)    #for RPF 2d: 384

    # setup core functions
    # case has the following functions which can be accessed as case.'function_name'
    # possible 'function_name': allocate, preprocess, allocate_eval, preprocess_eval, integrate, displacement, normalization_stats,
    
    case = case_builder(
        box=args.box,
        metadata=data_train.metadata,
        input_seq_length=args.config.input_seq_length,
        isotropic_norm=args.config.isotropic_norm,
        noise_std=args.config.noise_std,
        magnitude_features=args.config.magnitude_features,
        external_force_fn=data_train.external_force_fn,
        neighbor_list_backend=args.config.neighbor_list_backend,
        neighbor_list_multiplier=args.config.neighbor_list_multiplier,
        dtype=(jnp.float64 if args.config.f64 else jnp.float32),
    )

    _, particle_type = data_train[0]

    args.info.homogeneous_particles = particle_type.max() == particle_type.min() #True for RPF as there is only fluid particles
    args.metadata = data_train.metadata #metadata info from metadata.json
    args.normalization_stats = case.normalization_stats #mean and std of velocity and accleleration
    args.config.has_external_force = data_train.external_force_fn is not None #True for RPF

    # setup model from configs
    model, MODEL = setup_model(args)   #from experimments/utils.py GNS model is called
    model = hk.without_apply_rng(hk.transform_with_state(model)) #convert the 'model' function into haiku function

    # mixed precision training based on this reference:
    # https://github.com/deepmind/dm-haiku/blob/main/examples/imagenet/train.py
    policy = jmp.get_policy("params=float32,compute=float32,output=float32")
    hk.mixed_precision.set_policy(MODEL, policy)

    if args.config.mode == "train" or args.config.mode == "all":
        print("Start training...")
        # save config file
        run_prefix = f"{args.config.model}_{data_train.name}"
        data_and_time = datetime.today().strftime("%Y%m%d-%H%M%S")
        args.info.run_name = f"{run_prefix}_{data_and_time}"

        args.config.new_checkpoint = os.path.join(
            args.config.ckp_dir, args.info.run_name
        )
        os.makedirs(args.config.new_checkpoint, exist_ok=True)
        os.makedirs(os.path.join(args.config.new_checkpoint, "best"), exist_ok=True)
        with open(os.path.join(args.config.new_checkpoint, "config.yaml"), "w") as f:
            yaml.dump(vars(args.config), f)
        with open(
            os.path.join(args.config.new_checkpoint, "best", "config.yaml"), "w"
        ) as f:
            yaml.dump(vars(args.config), f)

        if args.config.wandb:
            # wandb doesn't like Namespace objects
            args_dict = copy.copy(args)
            args_dict.config = vars(args.config)
            args_dict.info = vars(args.info)

            wandb_run = wandb.init(
                project=args.config.wandb_project,
                entity=args.config.wandb_entity,
                name=args.info.run_name,
                config=args_dict,
                save_code=True,
            )
        else:
            wandb_run = None

        pf_config = PushforwardConfig(
            steps=args.config.pushforward["steps"],
            unrolls=args.config.pushforward["unrolls"],
            probs=args.config.pushforward["probs"],
        )

        trainer = Trainer(
            model,
            case,
            data_train, #data_train and data_eval extracted from setup_data() in experiments/utils.py
            data_eval,
            pushforward=pf_config,
            metrics=args.config.metrics,
            seed=args.config.seed,
            batch_size=args.config.batch_size,
            input_seq_length=args.config.input_seq_length,
            noise_std=args.config.noise_std,
            lr_start=args.config.lr_start,
            lr_final=args.config.lr_final,
            lr_decay_steps=args.config.lr_decay_steps,
            lr_decay_rate=args.config.lr_decay_rate,
            loss_weight=args.config.loss_weight,
            n_rollout_steps=args.config.n_rollout_steps,
            eval_n_trajs=args.config.eval_n_trajs,
            rollout_dir=args.config.rollout_dir,
            out_type=args.config.out_type,
            log_steps=args.config.log_steps,
            eval_steps=args.config.eval_steps,
            metrics_stride=args.config.metrics_stride,
            is_pde_refiner = args.config.is_pde_refiner,
            num_refinement_steps = args.config.num_refinement_steps,
            sigma_min = args.config.sigma_min,
        )
        _, _, _ = trainer(
            step_max=args.config.step_max,
            load_checkpoint=args.config.model_dir,
            store_checkpoint=args.config.new_checkpoint,
            wandb_run=wandb_run,
        )

        if args.config.wandb:
            wandb.finish()

    #Starting inference (on test data)
    if args.config.mode == "infer" or args.config.mode == "all":
        print("Start inference...")
        if args.config.mode == "all":
            args.config.test = True  #Set to True, to load test data for inference.
            data_train, data_eval, args = setup_data(args)

            args.config.model_dir = os.path.join(args.config.new_checkpoint, "best")
            assert osp.isfile(os.path.join(args.config.model_dir, "params_tree.pkl"))

            args.config.rollout_dir = args.config.model_dir.replace("ckp", "rollout")
            os.makedirs(args.config.rollout_dir, exist_ok=True)

            if args.config.eval_n_trajs_infer is None:
                args.config.eval_n_trajs_infer = args.config.eval_n_trajs

        assert args.config.model_dir, "model_dir must be specified for inference."
        
        if args.config.is_pde_refiner:
            
            metrics = infer_pde_refiner(
                model,
                case,
                data_eval, #data_eval extracted from setup_data() in experiments/utils.py
                load_checkpoint=args.config.model_dir, #check point directory
                metrics=args.config.metrics_infer,  #mse, sink_horn and e_kin
                rollout_dir=args.config.rollout_dir,
                eval_n_trajs=args.config.eval_n_trajs_infer, #=454
                n_rollout_steps=args.config.n_rollout_steps, #20
                out_type=args.config.out_type_infer, # out_type = 'pkl'
                n_extrap_steps=args.config.n_extrap_steps, #=0
                seed=args.config.seed,
                metrics_stride=args.config.metrics_stride_infer, #=1
                num_refinement_steps = args.config.num_refinement_steps,
                sigma_min = args.config.sigma_min,
            )

        else:
            metrics = infer(  #inside lagrangebench/evaluate/rollout.py
                model,
                case,
                data_eval, #data_eval extracted from setup_data() in experiments/utils.py
                load_checkpoint=args.config.model_dir, #check point directory
                metrics=args.config.metrics_infer,  #mse, sink_horn and e_kin
                rollout_dir=args.config.rollout_dir,
                eval_n_trajs=args.config.eval_n_trajs_infer, #=-1
                n_rollout_steps=args.config.n_rollout_steps, #20
                out_type=args.config.out_type_infer, # out_type = 'pkl'
                n_extrap_steps=args.config.n_extrap_steps, #=0
                seed=args.config.seed,
                metrics_stride=args.config.metrics_stride_infer, #=1
            )

        split = "test" if args.config.test else "valid"
        print(f"Metrics of {args.config.model_dir} on {split} split:")
        print(averaged_metrics(metrics))
