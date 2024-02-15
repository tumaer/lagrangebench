import os
import os.path as osp
from datetime import datetime

import haiku as hk
import jmp
import numpy as np
import wandb

from experiments.utils import setup_data, setup_model
from lagrangebench import Trainer, infer
from lagrangebench.case_setup import case_builder
from lagrangebench.config import cfg_to_dict
from lagrangebench.evaluate import averaged_metrics


def train_or_infer(cfg):
    mode = cfg.mode
    old_model_dir = cfg.model.model_dir
    is_test = cfg.eval.test

    data_train, data_valid, data_test, dataset_name = setup_data(cfg)

    exp_info = {"dataset_name": dataset_name}

    metadata = data_train.metadata
    # neighbors search
    bounds = np.array(metadata["bounds"])
    box = bounds[:, 1] - bounds[:, 0]

    exp_info["len_train"] = len(data_train)
    exp_info["len_eval"] = len(data_valid)

    # setup core functions
    case = case_builder(
        box=box,
        metadata=metadata,
        external_force_fn=data_train.external_force_fn,
    )

    _, particle_type = data_train[0]

    # setup model from configs
    model, MODEL = setup_model(
        cfg,
        metadata=metadata,
        homogeneous_particles=particle_type.max() == particle_type.min(),
        has_external_force=data_train.external_force_fn is not None,
        normalization_stats=case.normalization_stats,
    )
    model = hk.without_apply_rng(hk.transform_with_state(model))

    # mixed precision training based on this reference:
    # https://github.com/deepmind/dm-haiku/blob/main/examples/imagenet/train.py
    policy = jmp.get_policy("params=float32,compute=float32,output=float32")
    hk.mixed_precision.set_policy(MODEL, policy)

    if mode == "train" or mode == "all":
        print("Start training...")
        # save config file
        run_prefix = f"{cfg.model.name}_{data_train.name}"
        data_and_time = datetime.today().strftime("%Y%m%d-%H%M%S")
        exp_info["run_name"] = f"{run_prefix}_{data_and_time}"

        cfg.model.model_dir = os.path.join(cfg.logging.ckp_dir, exp_info["run_name"])
        os.makedirs(cfg.model.model_dir, exist_ok=True)
        os.makedirs(os.path.join(cfg.model.model_dir, "best"), exist_ok=True)
        with open(os.path.join(cfg.model.model_dir, "config.yaml"), "w") as f:
            cfg.dump(stream=f)
        with open(os.path.join(cfg.model.model_dir, "best", "config.yaml"), "w") as f:
            cfg.dump(stream=f)

        if cfg.logging.wandb:
            cfg_dict = cfg_to_dict(cfg)
            cfg_dict.update(exp_info)

            wandb_run = wandb.init(
                project=cfg.logging.wandb_project,
                entity=cfg.logging.wandb_entity,
                name=cfg.logging.run_name,
                config=cfg_dict,
                save_code=True,
            )
        else:
            wandb_run = None

        trainer = Trainer(model, case, data_train, data_valid)
        _, _, _ = trainer(
            step_max=cfg.train.step_max,
            load_checkpoint=old_model_dir,
            store_checkpoint=cfg.model.model_dir,
            wandb_run=wandb_run,
        )

        if cfg.logging.wandb:
            wandb.finish()

    if mode == "infer" or mode == "all":
        print("Start inference...")
        best_model_dir = old_model_dir
        if mode == "all":
            best_model_dir = os.path.join(cfg.model.model_dir, "best")
            assert osp.isfile(os.path.join(best_model_dir, "params_tree.pkl"))

            cfg.eval.rollout_dir = best_model_dir.replace("ckp", "rollout")
            os.makedirs(cfg.eval.rollout_dir, exist_ok=True)

            if cfg.eval.n_trajs_infer is None:
                cfg.eval.n_trajs_infer = cfg.eval.n_trajs_train

        assert old_model_dir, "model_dir must be specified for inference."
        metrics = infer(
            model,
            case,
            data_test if is_test else data_valid,
            load_checkpoint=best_model_dir,
        )

        split = "test" if is_test else "valid"
        print(f"Metrics of {best_model_dir} on {split} split:")
        print(averaged_metrics(metrics))
