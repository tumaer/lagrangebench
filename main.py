import argparse
import os


def cli_arguments():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)

    # config arguments
    group.add_argument("-c", "--config", type=str, help="Path to the config yaml.")
    group.add_argument("--model_dir", type=str, help="Path to the model checkpoint.")
    # misc arguments
    parser.add_argument(
        "--gpu", type=int, required=False, help="CUDA device ID to use."
    )
    parser.add_argument(
        "--f64",
        required=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to use double precision.",
    )
    parser.add_argument(
        "--xla_mem_fraction",
        type=float,
        required=False,
        default=0.7,
        help="Fraction of XLA memory to use.",
    )
    # optional config overrides
    parser.add_argument(
        "extra",
        default=None,
        nargs=argparse.REMAINDER,
        help="Extra config overrides as key value pairs.",
    )

    args = parser.parse_args()
    if args.extra is None:
        args.extra = []

    return args


if __name__ == "__main__":
    cli_args = cli_arguments()

    if cli_args.config is not None:  # to (re)start training
        config_path = cli_args.config.strip()
    elif cli_args.model_dir is not None:  # to run inference
        config_path = os.path.join(cli_args.model_dir, "config.yaml")
        cli_args.extra.extend(["model.model_dir", cli_args.model_dir])

    # specify cuda device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152 from TensorFlow
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cli_args.gpu)
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(cli_args.xla_mem_fraction)
    if cli_args.f64:
        from jax import config

        config.update("jax_enable_x64", True)
    else:
        cli_args.extra.extend(["dtype", "float32"])

    from lagrangebench.config import cfg, load_cfg

    load_cfg(cfg, config_path, cli_args.extra)

    print("#" * 79, "\nStarting a LagrangeBench run with the following configs:")
    print(cfg.dump())
    print("#" * 79)

    from experiments.run import train_or_infer

    train_or_infer(cfg)
