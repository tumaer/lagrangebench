"""Default lagrangebench configs."""


from omegaconf import DictConfig, OmegaConf


def set_defaults(cfg: DictConfig = OmegaConf.create({})) -> DictConfig:
    """Set default lagrangebench configs."""

    ### global and hardware-related configs

    # configuration file. Either "config" or "load_ckp" must be specified.
    # If "config" is specified, "load_ckp" is ignored.
    cfg.config = None
    # Load checkpointed model from this directory
    cfg.load_ckp = None
    # One of "train", "infer" or "all" (= both)
    cfg.mode = "all"
    # random seed
    cfg.seed = 0
    # data type for preprocessing. One of "float32" or "float64"
    cfg.dtype = "float64"
    # gpu device. -1 for CPU. Should be specified before importing the library.
    cfg.gpu = None
    # XLA memory fraction to be preallocated. The JAX default is 0.75.
    # Should be specified before importing the library.
    cfg.xla_mem_fraction = None

    ### dataset
    cfg.dataset = OmegaConf.create({})

    # path to data directory
    cfg.dataset.src = None
    # dataset name
    cfg.dataset.name = None

    ### model
    cfg.model = OmegaConf.create({})

    # model architecture name. gns, segnn, egnn
    cfg.model.name = None
    # Length of the position input sequence
    cfg.model.input_seq_length = 6
    # Number of message passing steps
    cfg.model.num_mp_steps = 10
    # Number of MLP layers
    cfg.model.num_mlp_layers = 2
    # Hidden dimension
    cfg.model.latent_dim = 128
    # whether to include velocity magnitude features
    cfg.model.magnitude_features = False
    #  whether to normalize dimensions equally
    cfg.model.isotropic_norm = False

    # SEGNN only parameters
    # steerable attributes level
    cfg.model.lmax_attributes = 1
    # Level of the hidden layer
    cfg.model.lmax_hidden = 1
    # SEGNN normalization. instance, batch, none
    cfg.model.segnn_norm = "none"
    # SEGNN velocity aggregation. avg or last
    cfg.model.velocity_aggregate = "avg"

    ### training
    cfg.train = OmegaConf.create({})

    # batch size
    cfg.train.batch_size = 1
    # max number of training steps
    cfg.train.step_max = 500_000
    # number of workers for data loading
    cfg.train.num_workers = 4
    # standard deviation of the GNS-style noise
    cfg.train.noise_std = 3.0e-4

    # optimizer
    cfg.train.optimizer = OmegaConf.create({})

    # initial learning rate
    cfg.train.optimizer.lr_start = 1.0e-4
    # final learning rate (after exponential decay)
    cfg.train.optimizer.lr_final = 1.0e-6
    # learning rate decay rate
    cfg.train.optimizer.lr_decay_rate = 0.1
    # number of steps to decay learning rate
    cfg.train.optimizer.lr_decay_steps = 1.0e5

    # pushforward
    cfg.train.pushforward = OmegaConf.create({})

    # At which training step to introduce next unroll stage
    cfg.train.pushforward.steps = [-1, 20000, 300000, 400000]
    # For how many steps to unroll
    cfg.train.pushforward.unrolls = [0, 1, 2, 3]
    # Which probability ratio to keep between the unrolls
    cfg.train.pushforward.probs = [18, 2, 1, 1]

    # loss weights
    cfg.train.loss_weight = OmegaConf.create({})

    # weight for acceleration error
    cfg.train.loss_weight.acc = 1.0
    # weight for velocity error
    cfg.train.loss_weight.vel = 0.0
    # weight for position error
    cfg.train.loss_weight.pos = 0.0

    ### evaluation
    cfg.eval = OmegaConf.create({})

    # number of eval rollout steps. -1 is full rollout
    cfg.eval.n_rollout_steps = 20
    # whether to use the test or valid split
    cfg.eval.test = False
    # rollouts directory
    cfg.eval.rollout_dir = None

    # configs for validation during training
    cfg.eval.train = OmegaConf.create({})

    # number of trajectories to evaluate
    cfg.eval.train.n_trajs = 50
    # stride for e_kin and sinkhorn
    cfg.eval.train.metrics_stride = 10
    # batch size
    cfg.eval.train.batch_size = 1
    # metrics to evaluate
    cfg.eval.train.metrics = ["mse"]
    # write validation rollouts. One of "none", "vtk", or "pkl"
    cfg.eval.train.out_type = "none"

    # configs for inference/testing
    cfg.eval.infer = OmegaConf.create({})

    # number of trajectories to evaluate during inference
    cfg.eval.infer.n_trajs = -1
    # stride for e_kin and sinkhorn
    cfg.eval.infer.metrics_stride = 1
    # batch size
    cfg.eval.infer.batch_size = 2
    # metrics for inference
    cfg.eval.infer.metrics = ["mse", "e_kin", "sinkhorn"]
    # write inference rollouts. One of "none", "vtk", or "pkl"
    cfg.eval.infer.out_type = "pkl"

    # number of extrapolation steps during inference
    cfg.eval.infer.n_extrap_steps = 0

    ### logging
    cfg.logging = OmegaConf.create({})

    # number of steps between loggings
    cfg.logging.log_steps = 1000
    # number of steps between evaluations and checkpoints
    cfg.logging.eval_steps = 10000
    # wandb enable
    cfg.logging.wandb = False
    # wandb project name
    cfg.logging.wandb_project = None
    # wandb entity name
    cfg.logging.wandb_entity = "lagrangebench"
    # checkpoint directory
    cfg.logging.ckp_dir = "ckp"
    # name of training run
    cfg.logging.run_name = None

    ### neighbor list
    cfg.neighbors = OmegaConf.create({})

    # backend for neighbor list computation
    cfg.neighbors.backend = "jaxmd_vmap"
    # multiplier for neighbor list capacity
    cfg.neighbors.multiplier = 1.25

    return cfg


defaults = set_defaults()


def check_cfg(cfg: DictConfig):
    """Check if the configs are valid."""

    assert cfg.mode in ["train", "infer", "all"]
    assert cfg.dtype in ["float32", "float64"]
    assert cfg.dataset.src is not None, "dataset.src must be specified."

    assert cfg.model.input_seq_length >= 2, "At least two positions for one past vel."

    pf = cfg.train.pushforward
    assert len(pf.steps) == len(pf.unrolls) == len(pf.probs)
    assert all([s >= 0 for s in pf.unrolls]), "All unrolls must be non-negative."
    assert all([s >= 0 for s in pf.probs]), "All probabilities must be non-negative."
    lwv = cfg.train.loss_weight.values()
    assert all([w >= 0 for w in lwv]), "All loss weights must be non-negative."
    assert sum(lwv) > 0, "At least one loss weight must be non-zero."

    assert cfg.eval.train.n_trajs >= -1
    assert cfg.eval.infer.n_trajs >= -1
    assert set(cfg.eval.train.metrics).issubset(["mse", "e_kin", "sinkhorn"])
    assert set(cfg.eval.infer.metrics).issubset(["mse", "e_kin", "sinkhorn"])
    assert cfg.eval.train.out_type in ["none", "vtk", "pkl"]
    assert cfg.eval.infer.out_type in ["none", "vtk", "pkl"]
