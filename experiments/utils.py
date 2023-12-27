import os
import os.path as osp
from argparse import Namespace
from typing import Callable, Tuple, Type

import jax
import jax.numpy as jnp
from e3nn_jax import Irreps
from jax_md import space

from lagrangebench import models
from lagrangebench.data import H5Dataset
from lagrangebench.models.utils import node_irreps
from lagrangebench.utils import NodeType


def setup_data(args: Namespace) -> Tuple[H5Dataset, H5Dataset, Namespace]:
    if not osp.isabs(args.config.data_dir):
        args.config.data_dir = osp.join(os.getcwd(), args.config.data_dir)

    args.info.dataset_name = osp.basename(args.config.data_dir.split("/")[-1])
    if args.config.ckp_dir is not None:
        os.makedirs(args.config.ckp_dir, exist_ok=True)
    if args.config.rollout_dir is not None:
        os.makedirs(args.config.rollout_dir, exist_ok=True)

    # dataloader
    data_train = H5Dataset(
        "train",
        dataset_path=args.config.data_dir,
        input_seq_length=args.config.input_seq_length,
        extra_seq_length=args.config.pushforward["unrolls"][-1],
        nl_backend=args.config.neighbor_list_backend,
    )
    data_valid = H5Dataset(
        "valid",
        dataset_path=args.config.data_dir,
        input_seq_length=args.config.input_seq_length,
        extra_seq_length=args.config.n_rollout_steps,
        nl_backend=args.config.neighbor_list_backend,
    )
    data_test = H5Dataset(
        "test",
        dataset_path=args.config.data_dir,
        input_seq_length=args.config.input_seq_length,
        extra_seq_length=args.config.n_rollout_steps,
        nl_backend=args.config.neighbor_list_backend,
    )
    if args.config.eval_n_trajs == -1:
        args.config.eval_n_trajs = data_valid.num_samples
    if args.config.eval_n_trajs_infer == -1:
        args.config.eval_n_trajs_infer = data_valid.num_samples
    assert data_valid.num_samples >= args.config.eval_n_trajs, (
        f"Number of available evaluation trajectories ({data_valid.num_samples}) "
        f"exceeds eval_n_trajs ({args.config.eval_n_trajs})"
    )

    # TODO: move this to a more suitable place
    if "RPF" in args.info.dataset_name.upper():
        args.info.has_external_force = True
        if data_train.metadata["dim"] == 2:

            def external_force_fn(position):
                return jnp.where(
                    position[1] > 1.0,
                    jnp.array([-1.0, 0.0]),
                    jnp.array([1.0, 0.0]),
                )

        elif data_train.metadata["dim"] == 3:

            def external_force_fn(position):
                return jnp.where(
                    position[1] > 1.0,
                    jnp.array([-1.0, 0.0, 0.0]),
                    jnp.array([1.0, 0.0, 0.0]),
                )

    else:
        args.info.has_external_force = False
        external_force_fn = None

    data_train.external_force_fn = external_force_fn
    data_valid.external_force_fn = external_force_fn
    data_test.external_force_fn = external_force_fn

    return data_train, data_valid, data_test, args


def setup_model(args: Namespace) -> Tuple[Callable, Type]:
    """Setup model based on args."""
    model_name = args.config.model.lower()
    metadata = args.metadata

    if model_name == "gns":

        def model_fn(x):
            return models.GNS(
                particle_dimension=metadata["dim"],
                latent_size=args.config.latent_dim,
                blocks_per_step=args.config.num_mlp_layers,
                num_mp_steps=args.config.num_mp_steps,
                num_particle_types=NodeType.SIZE,
                particle_type_embedding_size=16,
            )(x)

        MODEL = models.GNS
    elif model_name == "segnn":
        # Hx1o vel, Hx0e vel, 2x1o boundary, 9x0e type
        node_feature_irreps = node_irreps(
            metadata,
            args.config.input_seq_length,
            args.config.has_external_force,
            args.config.magnitude_features,
            args.info.homogeneous_particles,
        )
        # 1o displacement, 0e distance
        edge_feature_irreps = Irreps("1x1o + 1x0e")

        def model_fn(x):
            return models.SEGNN(
                node_features_irreps=node_feature_irreps,
                edge_features_irreps=edge_feature_irreps,
                scalar_units=args.config.latent_dim,
                lmax_hidden=args.config.lmax_hidden,
                lmax_attributes=args.config.lmax_attributes,
                output_irreps=Irreps("1x1o"),
                num_mp_steps=args.config.num_mp_steps,
                n_vels=args.config.input_seq_length - 1,
                velocity_aggregate=args.config.velocity_aggregate,
                homogeneous_particles=args.info.homogeneous_particles,
                blocks_per_step=args.config.num_mlp_layers,
                norm=args.config.segnn_norm,
            )(x)

        MODEL = models.SEGNN
    elif model_name == "egnn":
        box = args.box
        if jnp.array(metadata["periodic_boundary_conditions"]).any():
            displacement_fn, shift_fn = space.periodic(jnp.array(box))
        else:
            displacement_fn, shift_fn = space.free()

        displacement_fn = jax.vmap(displacement_fn, in_axes=(0, 0))
        shift_fn = jax.vmap(shift_fn, in_axes=(0, 0))

        def model_fn(x):
            return models.EGNN(
                hidden_size=args.config.latent_dim,
                output_size=1,
                dt=metadata["dt"] * metadata["write_every"],
                displacement_fn=displacement_fn,
                shift_fn=shift_fn,
                normalization_stats=args.normalization_stats,
                num_mp_steps=args.config.num_mp_steps,
                n_vels=args.config.input_seq_length - 1,
                residual=True,
            )(x)

        MODEL = models.EGNN
    elif model_name == "painn":
        assert args.config.magnitude_features, "PaiNN requires magnitudes"
        radius = metadata["default_connectivity_radius"] * 1.5

        def model_fn(x):
            return models.PaiNN(
                hidden_size=args.config.latent_dim,
                output_size=1,
                n_vels=args.config.input_seq_length - 1,
                radial_basis_fn=models.painn.gaussian_rbf(20, radius, trainable=True),
                cutoff_fn=models.painn.cosine_cutoff(radius),
                num_mp_steps=args.config.num_mp_steps,
            )(x)

        MODEL = models.PaiNN
    elif model_name == "linear":

        def model_fn(x):
            return models.Linear(dim_out=metadata["dim"])(x)

        MODEL = models.Linear

    return model_fn, MODEL
