import os
import os.path as osp
from datetime import datetime
from typing import Dict, Optional, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from e3nn_jax import Irreps
from jax import config
from jax_md import space
from omegaconf import DictConfig, OmegaConf

from lagrangebench import Trainer, infer, models
from lagrangebench.case_setup import case_builder
from lagrangebench.data import H5Dataset
from lagrangebench.defaults import check_cfg
from lagrangebench.evaluate import averaged_metrics
from lagrangebench.models.utils import (
    edge_feature_size,
    node_feature_size,
    node_irreps,
)
from lagrangebench.utils import NodeType


def train_or_infer(cfg: Union[Dict, DictConfig]):
    if isinstance(cfg, Dict):
        cfg = OmegaConf.create(cfg)
    # sanity check on the passed configs
    check_cfg(cfg)

    mode = cfg.mode
    load_ckp = cfg.load_ckp
    is_test = cfg.eval.test

    if cfg.dtype == "float64":
        config.update("jax_enable_x64", True)

    data_train, data_valid, data_test = setup_data(cfg)

    metadata = data_train.metadata
    # neighbors search
    bounds = np.array(metadata["bounds"])
    box = bounds[:, 1] - bounds[:, 0]

    # setup core functions
    case = case_builder(
        box=box,
        metadata=metadata,
        input_seq_length=cfg.model.input_seq_length,
        cfg_neighbors=cfg.neighbors,
        cfg_model=cfg.model,
        noise_std=cfg.train.noise_std,
        external_force_fn=data_train.external_force_fn,
        dtype=cfg.dtype,
    )

    _, particle_type = data_train[0]

    # setup model from configs
    model = setup_model(
        cfg,
        metadata=metadata,
        homogeneous_particles=particle_type.max() == particle_type.min(),
        has_external_force=data_train.external_force_fn is not None,
        normalization_stats=case.normalization_stats,
    )

    if mode == "train" or mode == "all":
        print("Start training...")

        if cfg.logging.run_name is None:
            run_prefix = f"{cfg.model.name}_{data_train.name}"
            data_and_time = datetime.today().strftime("%Y%m%d-%H%M%S")
            cfg.logging.run_name = f"{run_prefix}_{data_and_time}"

        store_ckp = os.path.join(cfg.logging.ckp_dir, cfg.logging.run_name)
        os.makedirs(store_ckp, exist_ok=True)
        os.makedirs(os.path.join(store_ckp, "best"), exist_ok=True)
        with open(os.path.join(store_ckp, "config.yaml"), "w") as f:
            OmegaConf.save(config=cfg, f=f.name)
        with open(os.path.join(store_ckp, "best", "config.yaml"), "w") as f:
            OmegaConf.save(config=cfg, f=f.name)

        # dictionary of configs which will be stored on W&B
        wandb_config = OmegaConf.to_container(cfg)

        trainer = Trainer(
            model,
            case,
            data_train,
            data_valid,
            cfg.train,
            cfg.eval,
            cfg.logging,
            input_seq_length=cfg.model.input_seq_length,
            seed=cfg.seed,
        )

        model, _ = trainer.train(
            step_max=cfg.train.step_max,
            load_ckp=load_ckp,
            store_ckp=store_ckp,
            wandb_config=wandb_config,
        )

    if mode == "infer" or mode == "all":
        print("Start inference...")

        if mode == "infer":
            model_dir = load_ckp
        if mode == "all":
            model_dir = os.path.join(store_ckp, "best")
            assert osp.isfile(os.path.join(model_dir, "model.eqx"))

            cfg.eval.rollout_dir = model_dir.replace("ckp", "rollout")
            os.makedirs(cfg.eval.rollout_dir, exist_ok=True)

            if cfg.eval.infer.n_trajs is None:
                cfg.eval.infer.n_trajs = cfg.eval.train.n_trajs

        assert model_dir, "model_dir must be specified for inference."
        metrics = infer(
            model,
            case,
            data_test if is_test else data_valid,
            load_ckp=model_dir,
            cfg_eval_infer=cfg.eval.infer,
            rollout_dir=cfg.eval.rollout_dir,
            n_rollout_steps=cfg.eval.n_rollout_steps,
            seed=cfg.seed,
        )

        split = "test" if is_test else "valid"
        print(f"Metrics of {model_dir} on {split} split:")
        print(averaged_metrics(metrics))

    return 0


def setup_data(cfg) -> Tuple[H5Dataset, H5Dataset, H5Dataset]:
    dataset_path = cfg.dataset.src
    dataset_name = cfg.dataset.name
    ckp_dir = cfg.logging.ckp_dir
    rollout_dir = cfg.eval.rollout_dir
    input_seq_length = cfg.model.input_seq_length
    n_rollout_steps = cfg.eval.n_rollout_steps
    nl_backend = cfg.neighbors.backend

    if not osp.isabs(dataset_path):
        dataset_path = osp.join(os.getcwd(), dataset_path)

    if ckp_dir is not None:
        os.makedirs(ckp_dir, exist_ok=True)
    if rollout_dir is not None:
        os.makedirs(rollout_dir, exist_ok=True)

    # dataloader
    data_train = H5Dataset(
        "train",
        dataset_path=dataset_path,
        name=dataset_name,
        input_seq_length=input_seq_length,
        extra_seq_length=cfg.train.pushforward.unrolls[-1],
        nl_backend=nl_backend,
    )
    data_valid = H5Dataset(
        "valid",
        dataset_path=dataset_path,
        name=dataset_name,
        input_seq_length=input_seq_length,
        extra_seq_length=n_rollout_steps,
        nl_backend=nl_backend,
    )
    data_test = H5Dataset(
        "test",
        dataset_path=dataset_path,
        name=dataset_name,
        input_seq_length=input_seq_length,
        extra_seq_length=n_rollout_steps,
        nl_backend=nl_backend,
    )

    return data_train, data_valid, data_test


def setup_model(
    cfg,
    metadata: Dict,
    homogeneous_particles: bool = False,
    has_external_force: bool = False,
    normalization_stats: Optional[Dict] = None,
) -> eqx.Module:
    """Setup model based on cfg."""
    model_name = cfg.model.name.lower()
    input_seq_length = cfg.model.input_seq_length
    magnitude_features = cfg.model.magnitude_features
    key = jax.random.PRNGKey(cfg.seed)

    if model_name == "gns":
        n_in = node_feature_size(
            metadata,
            input_seq_length,
            has_external_force=has_external_force,
            has_magnitudes=magnitude_features,
            has_homogeneous_particles=homogeneous_particles,
        )
        e_in = edge_feature_size(metadata)
        model = models.GNS(
            particle_dimension=metadata["dim"],
            latent_size=cfg.model.latent_dim,
            blocks_per_step=cfg.model.num_mlp_layers,
            num_mp_steps=cfg.model.num_mp_steps,
            num_particle_types=NodeType.SIZE,
            particle_type_embedding_size=16,
            node_in_size=n_in,
            edge_in_size=e_in,
            key=key,
        )
    elif model_name == "segnn":
        node_feature_irreps = node_irreps(
            metadata,
            input_seq_length,
            has_external_force,
            magnitude_features,
            homogeneous_particles,
        )
        edge_feature_irreps = Irreps("1x1o + 1x0e")
        model = models.SEGNN(
            node_features_irreps=node_feature_irreps,
            edge_features_irreps=edge_feature_irreps,
            scalar_units=cfg.model.latent_dim,
            lmax_hidden=cfg.model.lmax_hidden,
            lmax_attributes=cfg.model.lmax_attributes,
            output_irreps=Irreps("1x1o"),
            num_mp_steps=cfg.model.num_mp_steps,
            n_vels=input_seq_length - 1,
            velocity_aggregate=cfg.model.velocity_aggregate,
            homogeneous_particles=homogeneous_particles,
            blocks_per_step=cfg.model.num_mlp_layers,
            key=key,
        )
    elif model_name == "egnn":
        box = cfg.box
        if jnp.array(metadata["periodic_boundary_conditions"]).any():
            displacement_fn, shift_fn = space.periodic(jnp.array(box))
        else:
            displacement_fn, shift_fn = space.free()
        model = models.EGNN(
            hidden_size=cfg.model.latent_dim,
            output_size=1,
            dt=metadata["dt"] * metadata["write_every"],
            displacement_fn=displacement_fn,
            shift_fn=shift_fn,
            normalization_stats=normalization_stats,
            num_mp_steps=cfg.model.num_mp_steps,
            n_vels=input_seq_length - 1,
            homogeneous_particles=homogeneous_particles,
            residual=True,
            key=key,
        )
    elif model_name == "painn":
        assert magnitude_features, "PaiNN requires magnitudes"
        radius = metadata["default_connectivity_radius"] * 1.5
        n_rbf = 20
        # scalar input: vel_mag (input_seq_length - 1) + optional one-hot types
        s_in_size = (input_seq_length - 1) + (
            0 if homogeneous_particles else NodeType.SIZE
        )
        # vector input: traj (n_vels) + optional force (1) + optional bound (2)
        n_vels = input_seq_length - 1
        v_in_channels = n_vels
        if has_external_force:
            v_in_channels += 1
        if not any(metadata["periodic_boundary_conditions"]):
            v_in_channels += 2
        model = models.PaiNN(
            hidden_size=cfg.model.latent_dim,
            output_size=1,
            n_vels=n_vels,
            radial_basis_fn=models.painn.gaussian_rbf(n_rbf, radius, trainable=True),
            cutoff_fn=models.painn.cosine_cutoff(radius),
            num_mp_steps=cfg.model.num_mp_steps,
            n_rbf=n_rbf,
            s_in_size=s_in_size,
            v_in_channels=v_in_channels,
            homogeneous_particles=homogeneous_particles,
            key=key,
        )
    elif model_name == "linear":
        # input = vel_hist + vel_mag + bound + force + particle_type[:, None]
        n_in = node_feature_size(
            metadata,
            input_seq_length,
            has_external_force=has_external_force,
            has_magnitudes=magnitude_features,
            has_homogeneous_particles=homogeneous_particles,
        ) + 1
        model = models.Linear(in_features=n_in, dim_out=metadata["dim"], key=key)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model
