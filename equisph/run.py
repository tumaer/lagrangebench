import haiku as hk
import jax
import jmp
import numpy as np

from equisph.case_setup import case_builder
from equisph.data import get_dataset_stats, setup_data
from equisph.evaluate import MetricsComputer
from equisph.models import get_model
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
    case = case_builder(args, external_force_fn)

    # get an example to initialize the case and model
    pos_input_and_target, particle_type = next(iter(loader_train))
    # the torch loader give a whole batch. We take only the first sample.
    sample = (pos_input_and_target[0], particle_type[0])
    # initialize case
    key, features, _, neighbors = case.allocate(key, sample)

    args.info.homogeneous_particles = particle_type.max() == particle_type.min()
    # setup model from configs
    model, MODEL = get_model(args)
    # transform core simulator
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
        case.displacement,
        args.metadata,
        args.config.input_seq_length,
    )

    if args.config.mode == "train":
        from equisph.train import train

        train(
            model,
            case,
            params,
            state,
            neighbors,
            loader_train,
            loader_eval,
            metrics_computer,
            args,
        )
    elif args.config.mode == "infer":
        from equisph.evaluate import infer

        infer(
            model,
            case,
            params,
            state,
            neighbors,
            loader_eval,
            metrics_computer,
            args,
        )
