import os
import os.path as osp
from argparse import Namespace
from typing import Callable, Dict, Optional, Tuple, Type

import jax
import jax.numpy as jnp
from e3nn_jax import Irreps
from jax_md import space

from lagrangebench import models
from lagrangebench.data import H5Dataset
from lagrangebench.models.utils import node_irreps
from lagrangebench.utils import NodeType


def setup_data(cfg) -> Tuple[H5Dataset, H5Dataset, Namespace]:
    data_dir = cfg.data_dir
    ckp_dir = cfg.logging.ckp_dir
    rollout_dir = cfg.eval.rollout_dir
    input_seq_length = cfg.model.input_seq_length
    n_rollout_steps = cfg.eval.n_rollout_steps
    neighbor_list_backend = cfg.neighbors.backend
    if not osp.isabs(data_dir):
        data_dir = osp.join(os.getcwd(), data_dir)

    dataset_name = osp.basename(data_dir.split("/")[-1])
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
        nl_backend=neighbor_list_backend,
    )
    data_valid = H5Dataset(
        "valid",
        dataset_path=data_dir,
        input_seq_length=input_seq_length,
        extra_seq_length=n_rollout_steps,
        nl_backend=neighbor_list_backend,
    )
    data_test = H5Dataset(
        "test",
        dataset_path=data_dir,
        input_seq_length=input_seq_length,
        extra_seq_length=n_rollout_steps,
        nl_backend=neighbor_list_backend,
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

    return data_train, data_valid, data_test, dataset_name


def setup_model(
    cfg,
    metadata: Dict,
    homogeneous_particles: bool = False,
    has_external_force: bool = False,
    normalization_stats: Optional[Dict] = None,
) -> Tuple[Callable, Type]:
    """Setup model based on cfg."""
    model_name = cfg.model.name.lower()

    latent_dim = cfg.model.latent_dim
    num_mlp_layers = cfg.model.num_mlp_layers
    num_mp_steps = cfg.model.num_mp_steps

    input_seq_length = cfg.model.input_seq_length
    magnitude_features = cfg.train.magnitude_features

    if model_name == "gns":

        def model_fn(x):
            return models.GNS(
                particle_dimension=metadata["dim"],
                latent_size=latent_dim,
                blocks_per_step=num_mlp_layers,
                num_mp_steps=num_mp_steps,
                num_particle_types=NodeType.SIZE,
                particle_type_embedding_size=16,
            )(x)

        MODEL = models.GNS
    elif model_name == "segnn":
        segnn_cfg = cfg.model.segnn
        # Hx1o vel, Hx0e vel, 2x1o boundary, 9x0e type
        node_feature_irreps = node_irreps(
            metadata,
            input_seq_length,
            has_external_force,
            homogeneous_particles,
        )
        # 1o displacement, 0e distance
        edge_feature_irreps = Irreps("1x1o + 1x0e")

        def model_fn(x):
            return models.SEGNN(
                node_features_irreps=node_feature_irreps,
                edge_features_irreps=edge_feature_irreps,
                scalar_units=latent_dim,
                lmax_hidden=segnn_cfg.lmax_hidden,
                lmax_attributes=segnn_cfg.lmax_attributes,
                output_irreps=Irreps("1x1o"),
                num_mp_steps=num_mp_steps,
                n_vels=input_seq_length - 1,
                velocity_aggregate=segnn_cfg.velocity_aggregate,
                homogeneous_particles=cfg.train.homogeneous_particles,
                blocks_per_step=num_mlp_layers,
                norm=segnn_cfg.segnn_norm,
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
                hidden_size=cfg.latent_dim,
                output_size=1,
                dt=metadata["dt"] * metadata["write_every"],
                displacement_fn=displacement_fn,
                shift_fn=shift_fn,
                normalization_stats=normalization_stats,
                num_mp_steps=num_mp_steps,
                n_vels=input_seq_length - 1,
                residual=True,
            )(x)

        MODEL = models.EGNN
    elif model_name == "painn":
        assert magnitude_features, "PaiNN requires magnitudes"
        radius = metadata["default_connectivity_radius"] * 1.5

        def model_fn(x):
            return models.PaiNN(
                hidden_size=latent_dim,
                output_size=1,
                n_vels=input_seq_length - 1,
                radial_basis_fn=models.painn.gaussian_rbf(20, radius, trainable=True),
                cutoff_fn=models.painn.cosine_cutoff(radius),
                num_mp_steps=num_mp_steps,
            )(x)

        MODEL = models.PaiNN
    elif model_name == "linear":

        def model_fn(x):
            return models.Linear(dim_out=metadata["dim"])(x)

        MODEL = models.Linear

    return model_fn, MODEL
