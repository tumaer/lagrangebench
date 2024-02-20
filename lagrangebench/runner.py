import os
import os.path as osp
from argparse import Namespace
from datetime import datetime
from typing import Callable, Dict, Optional, Tuple, Type

import haiku as hk
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from e3nn_jax import Irreps
from jax_md import space

from lagrangebench import Trainer, infer, models
from lagrangebench.case_setup import case_builder
from lagrangebench.data import H5Dataset
from lagrangebench.evaluate import averaged_metrics
from lagrangebench.models.utils import node_irreps
from lagrangebench.utils import NodeType


def train_or_infer(cfg):
    mode = cfg.main.mode
    old_model_dir = cfg.model.model_dir
    is_test = cfg.eval.test

    data_train, data_valid, data_test = setup_data(cfg)

    metadata = data_train.metadata
    # neighbors search
    bounds = np.array(metadata["bounds"])
    box = bounds[:, 1] - bounds[:, 0]

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

        if cfg.logging.run_name is None:
            run_prefix = f"{cfg.model.name}_{data_train.name}"
            data_and_time = datetime.today().strftime("%Y%m%d-%H%M%S")
            cfg.logging.run_name = f"{run_prefix}_{data_and_time}"

        cfg.model.model_dir = os.path.join(cfg.logging.ckp_dir, cfg.logging.run_name)
        os.makedirs(cfg.model.model_dir, exist_ok=True)
        os.makedirs(os.path.join(cfg.model.model_dir, "best"), exist_ok=True)
        with open(os.path.join(cfg.model.model_dir, "config.yaml"), "w") as f:
            cfg.dump(stream=f)
        with open(os.path.join(cfg.model.model_dir, "best", "config.yaml"), "w") as f:
            cfg.dump(stream=f)

        trainer = Trainer(model, case, data_train, data_valid)
        _, _, _ = trainer(
            step_max=cfg.train.step_max,
            load_checkpoint=old_model_dir,
            store_checkpoint=cfg.model.model_dir,
        )

    if mode == "infer" or mode == "all":
        print("Start inference...")

        if mode == "infer":
            model_dir = cfg.model.model_dir
        if mode == "all":
            model_dir = os.path.join(cfg.model.model_dir, "best")
            assert osp.isfile(os.path.join(model_dir, "params_tree.pkl"))

            cfg.eval.rollout_dir = model_dir.replace("ckp", "rollout")
            os.makedirs(cfg.eval.rollout_dir, exist_ok=True)

            if cfg.eval.n_trajs_infer is None:
                cfg.eval.n_trajs_infer = cfg.eval.n_trajs_train

        assert model_dir, "model_dir must be specified for inference."
        metrics = infer(
            model,
            case,
            data_test if is_test else data_valid,
            load_checkpoint=model_dir,
        )

        split = "test" if is_test else "valid"
        print(f"Metrics of {model_dir} on {split} split:")
        print(averaged_metrics(metrics))


def setup_data(cfg) -> Tuple[H5Dataset, H5Dataset, Namespace]:
    data_dir = cfg.main.data_dir
    ckp_dir = cfg.logging.ckp_dir
    rollout_dir = cfg.eval.rollout_dir
    input_seq_length = cfg.model.input_seq_length
    n_rollout_steps = cfg.eval.n_rollout_steps
    if not osp.isabs(data_dir):
        data_dir = osp.join(os.getcwd(), data_dir)

    if ckp_dir is not None:
        os.makedirs(ckp_dir, exist_ok=True)
    if rollout_dir is not None:
        os.makedirs(rollout_dir, exist_ok=True)

    # dataloader
    data_train = H5Dataset(
        "train",
        dataset_path=data_dir,
        input_seq_length=input_seq_length,
        extra_seq_length=cfg.optimizer.pushforward.unrolls[-1],
    )
    data_valid = H5Dataset(
        "valid",
        dataset_path=data_dir,
        input_seq_length=input_seq_length,
        extra_seq_length=n_rollout_steps,
    )
    data_test = H5Dataset(
        "test",
        dataset_path=data_dir,
        input_seq_length=input_seq_length,
        extra_seq_length=n_rollout_steps,
    )

    # TODO find another way to set these
    if cfg.eval.n_trajs_train == -1:
        cfg.eval.n_trajs_train = data_valid.num_samples
    if cfg.eval.n_trajs_infer == -1:
        cfg.eval.n_trajs_infer = data_valid.num_samples

    assert data_valid.num_samples >= cfg.eval.n_trajs_train, (
        f"Number of available evaluation trajectories ({data_valid.num_samples}) "
        f"exceeds eval_n_trajs ({cfg.eval.n_trajs_train})"
    )

    return data_train, data_valid, data_test


def setup_model(
    cfg,
    metadata: Dict,
    homogeneous_particles: bool = False,
    has_external_force: bool = False,
    normalization_stats: Optional[Dict] = None,
) -> Tuple[Callable, Type]:
    """Setup model based on cfg."""
    model_name = cfg.model.name.lower()

    input_seq_length = cfg.model.input_seq_length
    magnitude_features = cfg.train.magnitude_features

    if model_name == "gns":

        def model_fn(x):
            return models.GNS(
                particle_dimension=metadata["dim"],
                num_particle_types=NodeType.SIZE,
                particle_type_embedding_size=16,
            )(x)

        MODEL = models.GNS
    elif model_name == "segnn":
        # Hx1o vel, Hx0e vel, 2x1o boundary, 9x0e type
        node_feature_irreps = node_irreps(
            metadata,
            input_seq_length,
            has_external_force,
            magnitude_features,
            homogeneous_particles,
        )
        # 1o displacement, 0e distance
        edge_feature_irreps = Irreps("1x1o + 1x0e")

        def model_fn(x):
            return models.SEGNN(
                node_features_irreps=node_feature_irreps,
                edge_features_irreps=edge_feature_irreps,
                output_irreps=Irreps("1x1o"),
                n_vels=input_seq_length - 1,
                homogeneous_particles=homogeneous_particles,
            )(x)

        MODEL = models.SEGNN
    elif model_name == "egnn":
        box = cfg.box
        if jnp.array(metadata["periodic_boundary_conditions"]).any():
            displacement_fn, shift_fn = space.periodic(jnp.array(box))
        else:
            displacement_fn, shift_fn = space.free()

        displacement_fn = jax.vmap(displacement_fn, in_axes=(0, 0))
        shift_fn = jax.vmap(shift_fn, in_axes=(0, 0))

        def model_fn(x):
            return models.EGNN(
                output_size=1,
                dt=metadata["dt"] * metadata["write_every"],
                displacement_fn=displacement_fn,
                shift_fn=shift_fn,
                normalization_stats=normalization_stats,
                n_vels=input_seq_length - 1,
                residual=True,
            )(x)

        MODEL = models.EGNN
    elif model_name == "painn":
        assert magnitude_features, "PaiNN requires magnitudes"
        radius = metadata["default_connectivity_radius"] * 1.5

        def model_fn(x):
            return models.PaiNN(
                output_size=1,
                n_vels=input_seq_length - 1,
                radial_basis_fn=models.painn.gaussian_rbf(20, radius, trainable=True),
                cutoff_fn=models.painn.cosine_cutoff(radius),
            )(x)

        MODEL = models.PaiNN
    elif model_name == "linear":

        def model_fn(x):
            return models.Linear(dim_out=metadata["dim"])(x)

        MODEL = models.Linear

    return model_fn, MODEL