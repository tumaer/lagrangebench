import haiku as hk
import jax
import jmp
import numpy as np

from equisph.case_setup import NodeType, case_builder
from equisph.data import get_dataset_stats, setup_data
from equisph.evaluate import MetricsComputer
from equisph.utils import get_num_params, load_haiku, set_seed


def train_or_infer(args):
    key, seed_worker, generator = set_seed(args.config.seed)

    args.metadata, loader_train, loader_eval, external_force_fn = setup_data(
        args, seed_worker, generator
    )
    # input and target mean and std for normalization
    args.normalization = get_dataset_stats(
        args.metadata, args.config.isotropic_norm, args.config.noise_std
    )
    # neighbors search
    bounds = np.array(args.metadata["bounds"])
    args.box = bounds[:, 1] - bounds[:, 0]

    args.info.len_train = len(loader_train.dataset)
    args.info.len_eval = len(loader_eval.dataset)

    # setup core functions
    scenario = case_builder(args, external_force_fn)

    # get an example to initialize the scenario and model
    pos_input_and_target, particle_type = next(iter(loader_train))
    # the torch loader give a whole batch. We take only the first sample.
    sample = (pos_input_and_target[0], particle_type[0])
    # initialize scenario
    key, features, _, neighbors = scenario.allocate(key, sample)

    args.info.homogeneous_particles = particle_type.max() == particle_type.min()

    if args.config.model == "gns":
        from equisph.models import GNS

        MODEL = GNS

        def model(x):
            return GNS(
                particle_dimension=args.metadata["dim"],
                latent_size=args.config.latent_dim,
                num_mlp_layers=args.config.num_mlp_layers,
                num_message_passing_steps=args.config.num_mp_steps,
                num_particle_types=NodeType.SIZE,
                particle_type_embedding_size=16,
            )(x)

    elif args.config.model == "linear":
        from equisph.models import Linear

        MODEL = Linear

        def model(x):
            return Linear(dim_out=3)(x)

    elif "segnn" in args.config.model:
        from e3nn_jax import Irreps

        from equisph.models.utils import node_irreps

        # Hx1o vel, Hx0e vel, 2x1o boundary, 9x0e type
        args.info.node_feature_irreps = node_irreps(args)
        # 1o displacement, 0e distance
        args.info.edge_feature_irreps = Irreps("1x1o + 1x0e")

        if args.config.model == "segnn":
            from equisph.models import SEGNN

            MODEL = SEGNN

            def model(x):
                return SEGNN(
                    node_features_irreps=Irreps(args.info.node_feature_irreps),
                    edge_features_irreps=Irreps(args.info.edge_feature_irreps),
                    scalar_units=args.config.latent_dim,
                    lmax_hidden=args.config.lmax_hidden,
                    lmax_attributes=args.config.lmax_attributes,
                    output_irreps=Irreps("1x1o"),
                    num_layers=args.config.num_mp_steps,
                    n_vels=args.config.input_seq_length - 1,
                    velocity_aggregate=args.config.velocity_aggregate,
                    homogeneous_particles=args.info.homogeneous_particles,
                    blocks_per_layer=args.config.num_mlp_layers,
                    norm=args.config.segnn_norm,
                )(x)

        elif args.config.model == "hae_segnn":
            from equisph.models import HAESEGNN

            MODEL = HAESEGNN

            def model(x):
                return HAESEGNN(
                    node_features_irreps=Irreps(args.info.node_feature_irreps),
                    edge_features_irreps=Irreps(args.info.edge_feature_irreps),
                    scalar_units=args.config.latent_dim,
                    lmax_hidden=args.config.lmax_hidden,
                    lmax_attributes=args.config.lmax_attributes,
                    output_irreps=Irreps("1x1o"),
                    num_layers=args.config.num_mp_steps,
                    n_vels=args.config.input_seq_length - 1,
                    velocity_aggregate=args.config.velocity_aggregate,
                    homogeneous_particles=args.info.homogeneous_particles,
                    blocks_per_layer=args.config.num_mlp_layers,
                    norm=args.config.segnn_norm,
                    right_attribute=args.config.right_attribute,
                    attribute_embedding_blocks=args.config.attribute_embedding_blocks,
                )(x)

    # transform core simulator outputting accelerations.
    model = hk.without_apply_rng(hk.transform_with_state(model))

    # mixed precision training based on this reference:
    # https://github.com/deepmind/dm-haiku/blob/main/examples/imagenet/train.py
    policy = jmp.get_policy("params=float32,compute=float32,output=float32")
    hk.mixed_precision.set_policy(MODEL, policy)

    # initialize model
    key, subkey = jax.random.split(key, 2)
    params, state = model.init(subkey, (features, particle_type[0]))

    args.info.num_params = get_num_params(params)
    print(f"Model with {args.info.num_params} parameters.")

    # load model from checkpoint if provided
    if args.config.model_dir:
        params, state, _, args.info.step_start = load_haiku(args.config.model_dir)
        assert get_num_params(params) == args.info.num_params, (
            "Model size mismatch. "
            f"{args.info.num_params} (expected) vs {get_num_params(params)} (actual)."
        )
    else:
        args.info.step_start = 0

    metrics_computer = MetricsComputer(
        args.config.metrics,
        scenario.displacement,
        args.metadata,
        args.config.input_seq_length,
    )

    if args.config.mode == "train":
        from equisph.train import train

        train(
            model,
            params,
            state,
            neighbors,
            loader_train,
            loader_eval,
            scenario,
            metrics_computer,
            args,
        )
    elif args.config.mode == "infer":
        from equisph.evaluate import infer

        infer(
            model,
            params,
            state,
            neighbors,
            loader_eval,
            scenario,
            metrics_computer,
            args,
        )
