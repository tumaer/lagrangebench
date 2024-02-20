import argparse
import os


def cli_arguments():
    """Inspired by https://stackoverflow.com/a/51686813"""
    parser = argparse.ArgumentParser()

    # config arguments
    parser.add_argument("-c", "--config", type=str, help="Path to the config yaml.")
    parser.add_argument(
        "extra",
        default=None,
        nargs="*",
        help="Extra, optional config overrides. Need to be separated from '--config' "
        "by the pseudo-argument '--'.",
    )

    args = parser.parse_args()
    if args.extra is None:
        args.extra = []
    args.extra = preprocess_extras(args.extra)

    return args


def preprocess_extras(extras):
    """Preprocess extras.

    args.extra can be in any of the following 6 formats:
    {"--","-",""}key{" ","="}value

    Here we clean up {"--", "-", "="} and split into key value pairs
    """

    temp = []
    for arg in extras:
        if arg.startswith("--"):  # remove preceding "--"
            arg = arg[2:]
        elif arg.startswith("-"):  # remove preceding "-"
            arg = arg[1:]
        temp += arg.split("=")  # split key value pairs

    return temp


def import_cfg(config_path, extras):
    """Import cfg without executing lagrangebench.__init__().

    Based on:
    https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("temp", "lagrangebench/config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cfg = module.cfg
    load_cfg = module.load_cfg
    load_cfg(cfg, config_path, extras)
    return cfg


if __name__ == "__main__":
    cli_args = cli_arguments()

    if cli_args.config is not None:  # start from config.yaml
        config_path = cli_args.config.strip()
    elif "model.model_dir" in cli_args.extra:  # start from a checkpoint
        model_dir = cli_args.extra[cli_args.extra.index("model.model_dir") + 1]
        config_path = os.path.join(model_dir, "config.yaml")

    # load cfg without executing lagrangebench.__init__() -> temporary cfg for cuda
    cfg = import_cfg(config_path, cli_args.extra)

    # specify cuda device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152 from TensorFlow
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.main.gpu)
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(cfg.main.xla_mem_fraction)
    if cfg.main.dtype == "float64":
        from jax import config

        config.update("jax_enable_x64", True)

    # load cfg once again, this time executing lagrangebench.__init__() -> global cfg
    from lagrangebench.config import cfg, load_cfg

    load_cfg(cfg, config_path, cli_args.extra)

    print("#" * 79, "\nStarting a LagrangeBench run with the following configs:")
    print(cfg.dump())
    print("#" * 79)

    from lagrangebench.runner import train_or_infer

    train_or_infer(cfg)
