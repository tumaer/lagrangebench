"""Hydra entry point for LagrangeBench training and inference.

Invocation::

    python main.py --config-path configs/tgv_2d --config-name gns gpu=6

Any key defined in :mod:`lagrangebench.defaults` can be overridden from the
CLI with the standard ``key=value`` syntax. Example::

    python main.py -cp configs/tgv_2d -cn gns gpu=6 train.step_max=20000

Resuming from a checkpoint is a plain override:

    python main.py -cp configs/tgv_2d -cn gns mode=infer load_ckp=runs/ckp/gns_tgv2d_xyz
"""

import os

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from lagrangebench.defaults import defaults as _lb_defaults

# Register the Python defaults under ``lagrangebench_defaults`` so that config
# files can pull them in via ``defaults: [/lagrangebench_defaults, _self_]``.
_cs = ConfigStore.instance()
_cs.store(name="lagrangebench_defaults", node=_lb_defaults, package="_global_")


@hydra.main(version_base=None, config_path=None, config_name=None)
def main(cfg: DictConfig) -> int:
    # ``gpu`` and ``xla_mem_fraction`` must be set before importing jax.
    gpu = cfg.get("gpu", -1)
    gpu = -1 if gpu is None else int(gpu)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see TensorFlow issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if gpu == -1:
        os.environ["JAX_PLATFORMS"] = "cpu"
    mem_fraction = cfg.get("xla_mem_fraction", 0.75)
    if mem_fraction is None:
        mem_fraction = 0.75
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(mem_fraction)

    print("#" * 79, "\nStarting a LagrangeBench run with the following configs:")
    print(OmegaConf.to_yaml(cfg))
    print("#" * 79)

    from lagrangebench.runner import train_or_infer

    return train_or_infer(cfg)


if __name__ == "__main__":
    main()
