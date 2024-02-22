import os

from omegaconf import DictConfig, OmegaConf


def load_embedded_configs(config_path: str, cli_args: DictConfig) -> DictConfig:
    """Loads all 'extends' embedded configs and merge them with the cli overwrites."""

    cfgs = [OmegaConf.load(config_path)]
    while "extends" in cfgs[0]:
        extends_path = cfgs[0]["extends"]
        del cfgs[0]["extends"]
        cfgs = [OmegaConf.load(extends_path)] + cfgs
    cfg = OmegaConf.merge(*cfgs, cli_args)
    return cfg


if __name__ == "__main__":
    cli_args = OmegaConf.from_cli()
    assert ("config" in cli_args.main) != (
        "model_dir" in cli_args.main
    ), "You must specify one of main.config or main.model_dir."

    if "config" in cli_args.main:  # start from config.yaml
        config_path = cli_args.main.config
    elif "model_dir" in cli_args.main:  # start from a checkpoint
        config_path = os.path.join(cli_args.main.model_dir, "config.yaml")

    # values that need to be specified before importing jax
    cli_args.main.gpu = cli_args.main.get("gpu", -1)
    cli_args.main.xla_mem_fraction = cli_args.main.get("xla_mem_fraction", 0.75)

    # specify cuda device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152 from TensorFlow
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cli_args.main.gpu)
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(cli_args.main.xla_mem_fraction)

    cfg = load_embedded_configs(config_path, cli_args)

    print("#" * 79, "\nStarting a LagrangeBench run with the following configs:")
    print(OmegaConf.to_yaml(cfg))
    print("#" * 79)

    from lagrangebench.runner import train_or_infer

    train_or_infer(cfg)
