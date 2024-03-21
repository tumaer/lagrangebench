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
from lagrangebench import (
    Trainer,
    infer,
    infer_acdm,
    infer_pde_refiner,
    infer_with_multiple_samples,
    infer_with_state_avg_at_every_step,
)
from lagrangebench.case_setup import case_builder
from lagrangebench.evaluate import averaged_metrics
from lagrangebench.utils import ACDMConfig, PushforwardConfig


def train_or_infer(args: Namespace):
    data_train, data_valid, data_test, args = setup_data(args)

    # neighbors search
    bounds = np.array(data_train.metadata["bounds"])
    args.box = bounds[:, 1] - bounds[:, 0]

    args.info.len_train = len(data_train)
    args.info.len_eval = len(data_valid)

    # setup core functions
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

    args.info.homogeneous_particles = particle_type.max() == particle_type.min()
    args.metadata = data_train.metadata
    args.normalization_stats = case.normalization_stats
    args.config.has_external_force = data_train.external_force_fn is not None

    # setup model from configs
    model, MODEL = setup_model(args)
    model = hk.without_apply_rng(hk.transform_with_state(model))

    # mixed precision training based on this reference:
    # https://github.com/deepmind/dm-haiku/blob/main/examples/imagenet/train.py
    policy = jmp.get_policy("params=float32,compute=float32,output=float32")
    hk.mixed_precision.set_policy(MODEL, policy)

    acdm_config = ACDMConfig(
        diffusionSteps=args.config.diffusion_steps,
        num_conditioning_steps=args.config.num_conditioning_steps,
        conditioning_parameter=args.config.conditioning_parameter,
    )

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
            data_train,
            data_valid,
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
            num_workers=args.config.num_workers,
            batch_size_infer=args.config.batch_size_infer,
            is_pde_refiner=args.config.is_pde_refiner,
            num_refinement_steps=args.config.num_refinement_steps,
            sigma_min=args.config.sigma_min,
            refinement_parameter=args.config.refinement_parameter,
            is_acdm=args.config.is_acdm,
            acdm_config=acdm_config,
        )
        _, _, _ = trainer(
            step_max=args.config.step_max,
            load_checkpoint=args.config.model_dir,
            store_checkpoint=args.config.new_checkpoint,
            wandb_run=wandb_run,
        )

        if args.config.wandb:
            wandb.finish()

    if args.config.mode == "infer" or args.config.mode == "all":
        print("Start inference...")
        if args.config.mode == "all":
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
                data_test if args.config.test else data_valid,
                load_checkpoint=args.config.model_dir,  # check point directory
                metrics=args.config.metrics_infer,  # mse, sink_horn and e_kin
                rollout_dir=args.config.rollout_dir,
                eval_n_trajs=args.config.eval_n_trajs_infer,  # =454
                n_rollout_steps=args.config.n_rollout_steps,  # 20
                out_type=args.config.out_type_infer,  # out_type = 'pkl'
                n_extrap_steps=args.config.n_extrap_steps,  # =0
                seed=args.config.seed,
                metrics_stride=args.config.metrics_stride_infer,  # =1
                batch_size=args.config.batch_size_infer,
                num_refinement_steps=args.config.num_refinement_steps,
                sigma_min=args.config.sigma_min,
                refinement_parameter=args.config.refinement_parameter,
            )

        elif args.config.is_acdm and not args.config.state_avg_at_every_step \
            and not args.config.different_samples_rollout:
            metrics = infer_acdm(
                model,
                case,
                data_test if args.config.test else data_valid,
                load_checkpoint=args.config.model_dir,
                metrics=args.config.metrics_infer,
                rollout_dir=args.config.rollout_dir,
                eval_n_trajs=args.config.eval_n_trajs_infer,
                n_rollout_steps=args.config.n_rollout_steps,
                out_type=args.config.out_type_infer,
                n_extrap_steps=args.config.n_extrap_steps,
                seed=args.config.seed,
                metrics_stride=args.config.metrics_stride_infer,
                batch_size=args.config.batch_size_infer,
                acdm_config=acdm_config,
                noise_std=args.config.noise_std,
                input_seq_length=args.config.input_seq_length,
            )

        elif args.config.is_acdm and args.config.state_avg_at_every_step \
            and not args.config.different_samples_rollout:
            metrics = infer_with_state_avg_at_every_step(
                model,
                case,
                data_test if args.config.test else data_valid,
                load_checkpoint=args.config.model_dir,
                metrics=args.config.metrics_infer,
                rollout_dir=args.config.rollout_dir,
                eval_n_trajs=args.config.eval_n_trajs_infer,
                n_rollout_steps=args.config.n_rollout_steps,
                out_type=args.config.out_type_infer,
                n_extrap_steps=args.config.n_extrap_steps,
                seed=args.config.seed,
                metrics_stride=args.config.metrics_stride_infer,
                batch_size=args.config.batch_size_infer,
                acdm_config=acdm_config,
                noise_std=args.config.noise_std,
                input_seq_length=args.config.input_seq_length,
            )

        elif args.config.is_acdm and args.config.different_samples_rollout and not args.config.state_avg_at_every_step:
            metrics = infer_with_multiple_samples(
                model,
                case,
                data_test if args.config.test else data_valid,
                load_checkpoint=args.config.model_dir,
                metrics=args.config.metrics_infer,
                rollout_dir=args.config.rollout_dir,
                eval_n_trajs=args.config.eval_n_trajs_infer,
                n_rollout_steps=args.config.n_rollout_steps,
                out_type=args.config.out_type_infer,
                n_extrap_steps=args.config.n_extrap_steps,
                seed=args.config.seed,
                metrics_stride=args.config.metrics_stride_infer,
                batch_size=args.config.batch_size_infer,
                acdm_config=acdm_config,
                noise_std=args.config.noise_std,
                input_seq_length=args.config.input_seq_length,
            )

        else:
            metrics = infer(
                model,
                case,
                data_test if args.config.test else data_valid,
                load_checkpoint=args.config.model_dir,
                metrics=args.config.metrics_infer,
                rollout_dir=args.config.rollout_dir,
                eval_n_trajs=args.config.eval_n_trajs_infer,
                n_rollout_steps=args.config.n_rollout_steps,
                out_type=args.config.out_type_infer,
                n_extrap_steps=args.config.n_extrap_steps,
                seed=args.config.seed,
                metrics_stride=args.config.metrics_stride_infer,
                batch_size=args.config.batch_size_infer,
            )

        split = "test" if args.config.test else "valid"
        print(f"Metrics of {args.config.model_dir} on {split} split:")
        print(averaged_metrics(metrics))
