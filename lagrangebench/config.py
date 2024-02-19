from typing import Dict, List, Optional

import yaml
from yacs.config import CfgNode as CN

# lagrangebench-wide config object
cfg = CN()


def custom_config(fn):
    """Decorator to add custom config functions."""
    fn(cfg)
    return fn


def defaults(cfg):
    """Default lagrangebench values."""

    if cfg is None:
        raise ValueError("cfg should be a yacs CfgNode")

    # random seed
    cfg.seed = 0
    # data type for preprocessing
    cfg.dtype = "float64"

    # data directory
    cfg.data_dir = None
    # run, evaluation or both
    cfg.mode = "all"

    # model
    model = CN()

    model.name = None
    # Length of the position input sequence
    model.input_seq_length = 6
    # Number of message passing steps
    model.num_mp_steps = 10
    # Number of MLP layers
    model.num_mlp_layers = 2
    # Hidden dimension
    model.latent_dim = 128
    # Load checkpointed model from this directory
    model.model_dir = None

    cfg.model = model

    # training
    train = CN()

    # batch size
    train.batch_size = 1
    # max number of training steps
    train.step_max = 500_000
    # whether to include velocity magnitude features
    train.magnitude_features = False
    #  whether to normalize dimensions equally
    train.isotropic_norm = False
    # number of workers for data loading
    train.num_workers = 4

    cfg.train = train

    # optimizer
    optimizer = CN()

    # initial learning rate
    optimizer.lr_start = 1e-4
    # final learning rate (after exponential decay)
    optimizer.lr_final = 1e-6
    # learning rate decay rate
    optimizer.lr_decay_rate = 0.1
    # number of steps to decay learning rate
    optimizer.lr_decay_steps = 1e5
    # standard deviation of the GNS-style noise
    optimizer.noise_std = 3e-4

    # optimizer: pushforward
    pushforward = CN()
    # At which training step to introduce next unroll stage
    pushforward.steps = [-1, 20000, 300000, 400000]
    # For how many steps to unroll
    pushforward.unrolls = [0, 1, 2, 3]
    # Which probability ratio to keep between the unrolls
    pushforward.probs = [18, 2, 1, 1]

    # optimizer: loss weights
    loss_weight = CN()
    # weight for acceleration error
    loss_weight.acc = 1.0
    # weight for velocity error
    loss_weight.vel = 0.0
    # weight for position error
    loss_weight.pos = 0.0

    cfg.optimizer = optimizer
    cfg.optimizer.loss_weight = loss_weight
    cfg.optimizer.pushforward = pushforward

    # evaluation
    eval = CN()

    # number of eval rollout steps. -1 is full rollout
    eval.n_rollout_steps = 20
    # number of trajectories to evaluate during training
    eval.n_trajs_train = 1
    # number of trajectories to evaluate during inference
    eval.n_trajs_infer = 50
    # metrics for training
    eval.metrics_train = ["mse"]
    # stride for e_kin and sinkhorn
    eval.metrics_stride_train = 10
    # metrics for inference
    eval.metrics_infer = ["mse", "e_kin", "sinkhorn"]
    # stride for e_kin and sinkhorn
    eval.metrics_stride_infer = 1
    # number of extrapolation steps in inference
    eval.n_extrap_steps = 0
    # batch size for validation/testing
    eval.batch_size_infer = 2
    # write validation rollouts. One of "none", "vtk", or "pkl"
    eval.out_type_train = "none"
    # write inference rollouts. One of "none", "vtk", or "pkl"
    eval.out_type_infer = "pkl"
    # rollouts directory
    eval.rollout_dir = None
    # whether to use the test split
    eval.test = False

    cfg.eval = eval

    # logging
    logging = CN()

    # number of steps between loggings
    logging.log_steps = 1000
    # number of steps between evaluations and checkpoints
    logging.eval_steps = 10000
    # wandb enable
    logging.wandb = False
    # wandb project name
    logging.wandb_project = None
    # wandb entity name
    logging.wandb_entity = "lagrangebench"
    # checkpoint directory
    logging.ckp_dir = "ckp"
    # name of training run
    logging.run_name = None

    cfg.logging = logging

    # neighbor list
    neighbors = CN()

    # backend for neighbor list computation
    neighbors.backend = "jaxmd_vmap"
    # multiplier for neighbor list capacity
    neighbors.multiplier = 1.25

    cfg.neighbors = neighbors


def check_cfg(cfg):
    assert cfg.data_dir is not None, "cfg.data_dir must be specified."
    assert (
        cfg.train.step_max is not None and cfg.train.step_max > 0
    ), "cfg.train.step_max must be specified and larger than 0."


def load_cfg(cfg: CN, config_path: str, extra_args: Optional[List] = None):
    if cfg is None:
        raise ValueError("cfg should be a yacs CfgNode")
    if len(cfg) == 0:
        defaults(cfg)
    cfg.merge_from_file(config_path)
    if extra_args is not None:
        cfg.merge_from_list(extra_args)
    check_cfg(cfg)


def cfg_to_dict(cfg: CN) -> Dict:
    return yaml.safe_load(cfg.dump())


# TODO find a better way
defaults(cfg)
