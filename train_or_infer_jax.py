"""GNS in JAX with Haiku, Jraph, and JAX-MD"""

import copy
import json
import os
from typing import Dict, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jmp
import jraph
import numpy as np
import optax
import yaml
from jax import vmap
from torch.utils.data import DataLoader

import wandb
from gns_jax.data import H5Dataset, numpy_collate
from gns_jax.utils import (
    Linear,
    NodeType,
    averaged_metrics,
    broadcast_from_batch,
    broadcast_to_batch,
    eval_rollout,
    get_dataset_normalization,
    get_kinematic_mask,
    get_num_params,
    load_haiku,
    log_norm_fn,
    oversmooth_norm,
    push_forward_build,
    push_forward_sample_steps,
    save_haiku,
    set_seed_from_config,
    setup_builder,
)


def train(
    model,
    params,
    state,
    neighbors,
    loader_train,
    loader_valid,
    setup,
    graph_preprocess,
    args,
):

    # checkpointing and logging
    run_prefix = "_".join([args.config.model, args.info.dataset_name, ""])
    i = 0
    while os.path.isdir(os.path.join(args.config.ckp_dir, run_prefix + str(i))):
        i += 1
    args.info.run_name = run_prefix + str(i)

    ckp_dir = os.path.join(args.config.ckp_dir, args.info.run_name)
    os.makedirs(ckp_dir, exist_ok=True)
    os.makedirs(os.path.join(ckp_dir, "best"), exist_ok=True)

    # save config file
    with open(os.path.join(ckp_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(args.config), f)

    # wandb doesn't like Namespace objects
    args_ = copy.copy(args)
    args_.config = vars(args.config)
    args_.info = vars(args.info)

    if args.config.wandb:
        wandb.init(
            project=args.config.wandb_project,
            entity="segnn-sph",
            name=args.info.run_name,
            config=args_,
            save_code=True,
        )

    # Set learning rate to decay from 1e-4 to 1e-6 over 10M steps exponentially
    # and then keep it at 1e-6
    lr_scheduler = optax.exponential_decay(
        init_value=args.config.lr_start,
        transition_steps=int(5e6),
        decay_rate=args.config.lr_decay_rate,
        end_value=args.config.lr_final,
    )
    opt_init, opt_update = optax.adamw(learning_rate=lr_scheduler, weight_decay=1e-8)
    # continue training from checkpoint or initialize optimizer state
    if args.config.model_dir:
        _, _, opt_state, _ = load_haiku(args.config.model_dir)
    else:
        opt_state = opt_init(params)

    # Precompile model for evaluation
    model_apply = jax.jit(model.apply)

    def loss_fn(
        params: hk.Params,
        state: hk.State,
        features: Dict[str, jnp.ndarray],
        target: jnp.ndarray,
        particle_type: jnp.ndarray,
    ):
        graph = graph_preprocess(features, particle_type)

        # TODO oversmooth for SEGNN (now fails because of graph.graph)
        if args.config.oversmooth_norm_hops > 0:
            graph, most_recent_vel_magnitude = oversmooth_norm(
                graph, args.config.oversmooth_norm_hops, args.config.input_seq_length
            )

        pred, state = model.apply(params, state, graph)
        assert target.shape == pred.shape

        if args.config.oversmooth_norm_hops > 0:
            pred *= most_recent_vel_magnitude[:, None]

        non_kinematic_mask = jnp.logical_not(get_kinematic_mask(particle_type))
        num_non_kinematic = non_kinematic_mask.sum()

        if args.config.log_norm in ["output", "both"]:
            pred = log_norm_fn(pred)
            target = log_norm_fn(target)

        # MSE loss
        loss = ((pred - target) ** 2).sum(axis=-1)
        loss = jnp.where(non_kinematic_mask, loss, 0)
        loss = loss.sum() / num_non_kinematic

        return loss, state

    @jax.jit
    def train_step(
        params: hk.Module,
        state: hk.State,
        features_batch: Tuple[jraph.GraphsTuple, ...],
        target_batch: Tuple[jnp.ndarray, ...],
        particle_type_batch: Tuple[jnp.ndarray, ...],
        opt_state: optax.OptState,
    ) -> Tuple[float, hk.Params, hk.State, optax.OptState]:

        value_and_grad_vmap = vmap(
            jax.value_and_grad(loss_fn, has_aux=True), in_axes=(None, None, 0, 0, 0)
        )
        (loss, state), grads = value_and_grad_vmap(
            params, state, features_batch, target_batch, particle_type_batch
        )

        # aggregate over the first (batch) dimension of each leave element
        grads = jax.tree_map(lambda x: x.sum(axis=0), grads)
        state = jax.tree_map(lambda x: x.sum(axis=0), state)
        loss = jax.tree_map(lambda x: x.mean(axis=0), loss)

        updates, opt_state = opt_update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return loss, new_params, state, opt_state

    preprocess_vmap = jax.vmap(setup.preprocess, in_axes=(0, 0, None, 0, None))

    push_forward = push_forward_build(graph_preprocess, model_apply, setup)
    push_forward_vmap = jax.vmap(push_forward, in_axes=(0, 0, 0, 0, None, None))

    # prepare for batch training.
    key = jax.random.PRNGKey(args.config.seed)
    keys = jax.random.split(key, args.config.batch_size)
    neighbors_batch = broadcast_to_batch(neighbors, args.config.batch_size)

    step_digits = len(str(int(args.config.step_max)))
    step = args.info.step_start
    while step < args.config.step_max:

        for raw_batch in loader_train:
            # pos_input_and_target, particle_type = raw_batch

            key, unroll_steps = push_forward_sample_steps(
                key, step, args.config.pushforward
            )

            # The target computation incorporates the sampled number pushforward steps
            keys, features_batch, target_batch, neighbors_batch = preprocess_vmap(
                keys, raw_batch, args.config.noise_std, neighbors_batch, unroll_steps
            )

            # unroll for push-forward steps
            _current_pos = raw_batch[0][:, :, : args.config.input_seq_length]
            for _ in range(unroll_steps):
                _current_pos, neighbors_batch, features_batch = push_forward_vmap(
                    features_batch,
                    _current_pos,
                    raw_batch[1],
                    neighbors_batch,
                    params,
                    state,
                )

            if neighbors_batch.did_buffer_overflow.sum() > 0:
                # check if the neighbor list is too small for any of the
                # samples in the batch. If so, reallocate the neighbor list
                ind = np.argmax(neighbors_batch.did_buffer_overflow)
                edges_ = neighbors_batch.idx[ind].shape
                print(f"Reallocate neighbors list {edges_} at step {step}")
                sample = broadcast_from_batch(raw_batch, index=ind)
                _, _, _, nbrs = setup.allocate(keys[0], sample)
                print(f"To list {nbrs.idx.shape}")

                neighbors_batch = broadcast_to_batch(nbrs, args.config.batch_size)

                # To run the loop N times even if sometimes
                # did_buffer_overflow > 0 we directly return to the beginning
                continue

            loss, params, state, opt_state = train_step(
                params=params,
                state=state,
                features_batch=features_batch,
                target_batch=target_batch["acc"],
                particle_type_batch=raw_batch[1],
                opt_state=opt_state,
            )

            if step % args.config.log_steps == 0:
                if args.config.wandb:
                    wandb.log({"train/loss": loss.item()}, step)
                else:
                    step_str = str(step).zfill(step_digits)
                    print(f"{step_str}, train/loss: {loss.item():.5f}.")

            if step % args.config.eval_steps == 0 and step > 0:
                nbrs = broadcast_from_batch(neighbors_batch, index=0)
                eval_metrics, nbrs = eval_rollout(
                    setup=setup,
                    model_apply=model_apply,
                    params=params,
                    state=state,
                    neighbors=nbrs,
                    loader_valid=loader_valid,
                    num_rollout_steps=args.config.num_rollout_steps
                    + 1,  # +1 not training
                    num_trajs=args.config.eval_num_trajs,
                    rollout_dir=args.config.rollout_dir,
                    graph_preprocess=graph_preprocess,
                    out_type=args.config.out_type,
                    oversmooth_norm_hops=args.config.oversmooth_norm_hops,
                )
                # In the beginning of training, the dynamics ae very random and
                # we don"t want to influence the training neighbors list by
                # this extreme case of the dynamics. Therefore, we only update
                # the neighbors list inside of the eval_rollout function.
                # neighbors_batch = broadcast_to_batch(nbrs, args.batch_size)

                # TODO: is disabling the "save_step" argument a good idea?
                # if step % args.config.save_steps == 0:
                metrics = averaged_metrics(eval_metrics)
                metadata_ckp = {
                    "step": step,
                    "loss": metrics["val/loss"],
                }
                save_haiku(ckp_dir, params, state, opt_state, metadata_ckp)

                if args.config.wandb:
                    wandb.log(metrics, step)
                else:
                    print(metrics)

            step += 1
            if step == args.config.step_max:
                break


def infer(model, params, state, neighbors, loader_valid, setup, graph_preprocess, args):

    model_apply = jax.jit(model.apply)
    eval_metrics, _ = eval_rollout(
        setup=setup,
        model_apply=model_apply,
        params=params,
        state=state,
        neighbors=neighbors,
        loader_valid=loader_valid,
        num_rollout_steps=args.config.num_rollout_steps
        + 1,  # +1 because we are not training
        num_trajs=args.config.eval_num_trajs,
        rollout_dir=args.config.rollout_dir,
        graph_preprocess=graph_preprocess,
        out_type=args.config.out_type,
        eval_n_more_steps=args.config.eval_n_more_steps,
        oversmooth_norm_hops=args.config.oversmooth_norm_hops,
    )
    print(averaged_metrics(eval_metrics))


def run(args):

    seed_worker, generator = set_seed_from_config(args.config.seed)

    args.info.dataset_name = os.path.basename(args.config.data_dir.split("/")[-1])
    if args.config.ckp_dir is not None:
        os.makedirs(args.config.ckp_dir, exist_ok=True)
    if args.config.rollout_dir is not None:
        os.makedirs(args.config.rollout_dir, exist_ok=True)
    with open(os.path.join(args.config.data_dir, "metadata.json"), "r") as f:
        args.metadata = json.loads(f.read())

    args.normalization = get_dataset_normalization(
        args.metadata, args.config.isotropic_norm, args.config.noise_std
    )

    if args.config.num_rollout_steps == -1:
        args.config.num_rollout_steps = (
            args.metadata["sequence_length"] - args.config.input_seq_length
        )

    # dataloader
    train_seq_l = args.config.input_seq_length + args.config.pushforward["unrolls"][-1]
    data_train = H5Dataset(args.config.data_dir, "train", train_seq_l, is_rollout=False)
    loader_train = DataLoader(
        dataset=data_train,
        batch_size=args.config.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=numpy_collate,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    infer_split = "test" if args.config.test else "valid"
    data_valid = H5Dataset(
        args.config.data_dir, infer_split, args.config.input_seq_length, is_rollout=True
    )
    loader_valid = DataLoader(
        dataset=data_valid,
        batch_size=1,
        collate_fn=numpy_collate,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    args.info.len_train = len(data_train)
    args.info.len_valid = len(data_valid)

    # neighbors search
    bounds = np.array(args.metadata["bounds"])
    args.box = bounds[:, 1] - bounds[:, 0]

    # dataset-specific setup
    if args.info.dataset_name in ["BoxBath", "BoxBathSample"]:
        particle_dimension = 3
        # node_in = 37, edge_in = 4
        args.box *= 1.2
        args.info.has_external_force = True
        external_force_fn = lambda x: jnp.array([0.0, 0.0, -1.0])
    elif args.info.dataset_name in ["WaterDrop", "WaterDropSample"]:
        particle_dimension = 2
        # node_in = 30, edge_in = 3
        args.info.has_external_force = False
        external_force_fn = None
    elif "TGV" in args.info.dataset_name.upper():
        particle_dimension = 3
        args.info.has_external_force = False
        external_force_fn = None
    elif "RPF" in args.info.dataset_name:
        particle_dimension = 3
        args.info.has_external_force = True

        def external_force_fn(position):
            return jnp.where(
                position[1] > 1.0,
                jnp.array([-1.0, 0.0, 0.0]),
                jnp.array([1.0, 0.0, 0.0]),
            )

    elif "Hook" in args.info.dataset_name:
        particle_dimension = 3
        args.info.has_external_force = False
        external_force_fn = None

    # preprocessing allocate and update functions in the spirit of jax-md"s
    # `partition.neighbor_list`. And an integration utility with PBC.
    setup = setup_builder(args, external_force_fn)

    # first PRNG key
    key = jax.random.PRNGKey(args.config.seed)
    # get an example to initialize the setup and model
    pos_input_and_target, particle_type = next(iter(loader_train))
    # the torch loader give a whole batch. We take only the first sample.
    sample = (pos_input_and_target[0], particle_type[0])
    # initialize setup
    key, features, _, neighbors = setup.allocate(key, sample)
    # initialize model
    key, subkey = jax.random.split(key, 2)

    args.info.homogeneous_particles = particle_type.max() == particle_type.min()

    if args.config.model == "gns":
        from gns_jax.gns import GNS
        from gns_jax.utils import gns_graph_transform_builder

        MODEL = GNS
        model = lambda x: GNS(
            particle_dimension=particle_dimension,
            latent_size=args.config.latent_dim,
            num_mlp_layers=args.config.num_mlp_layers,
            num_message_passing_steps=args.config.num_mp_steps,
            num_particle_types=NodeType.SIZE,
            particle_type_embedding_size=16,
        )(x)
        graph_preprocess = gns_graph_transform_builder()

    elif args.config.model == "lin":
        MODEL = Linear
        model = lambda x: Linear(dim_out=3)(x)
        graph_preprocess = lambda f: [
            f[k] for k in ["vel_hist", "vel_mag", "bound", "force"] if k in f
        ]

    elif "segnn" in args.config.model:
        from e3nn_jax import Irreps
        from segnn_jax import SEGNN, weight_balanced_irreps

        from segnn_experiments.utils import node_irreps, segnn_graph_transform_builder

        hidden_irreps = weight_balanced_irreps(
            scalar_units=args.config.latent_dim,
            # attribute irreps
            irreps_right=Irreps.spherical_harmonics(args.config.lmax_attributes),
            use_sh=True,
            lmax=args.config.lmax_hidden,
        )
        if args.config.model == "segnn":
            MODEL = SEGNN
            model = lambda x: SEGNN(
                hidden_irreps=hidden_irreps,
                output_irreps=Irreps("1x1o"),
                num_layers=args.config.num_mp_steps,
                task="node",
                blocks_per_layer=args.config.num_mlp_layers,
                norm=args.config.segnn_norm,
            )(x)
        elif args.config.model == "segnn_rewind":
            from segnn_experiments.rsegnn import RSEGNN

            MODEL = RSEGNN
            assert (
                args.config.num_mp_steps == args.config.input_seq_length - 1
            ), "The number of layers must be the same size of the history"
            assert (
                args.config.velocity_aggregate == "all"
            ), "SEGNN with rewind is supposed to have all historical velocities"

            model = lambda x: RSEGNN(
                hidden_irreps=hidden_irreps,
                output_irreps=Irreps("1x1o"),
                num_layers=args.config.num_mp_steps,
                task="node",
                blocks_per_layer=args.config.num_mlp_layers,
                norm=args.config.segnn_norm,
            )(x)
        elif args.config.model == "segnn_attention":
            from segnn_experiments.asegnn import AttentionSEGNN

            MODEL = AttentionSEGNN
            assert (
                args.config.velocity_aggregate == "all"
            ), "SEGNN with attention is supposed to have all historical velocities"

            model = lambda x: AttentionSEGNN(
                hidden_irreps=hidden_irreps,
                output_irreps=Irreps("1x1o"),
                num_layers=args.config.num_mp_steps,
                lmax_latent=args.config.lmax_attributes,
                task="node",
                blocks_per_layer=args.config.num_mlp_layers,
                norm=args.config.segnn_norm,
                right_attribute=args.config.right_attribute,
                attribute_embedding_blocks=args.config.attention_blocks,
            )(x)

        args.info.node_feature_irreps = node_irreps(args)

        graph_preprocess = segnn_graph_transform_builder(
            node_features_irreps=Irreps(
                args.info.node_feature_irreps
            ),  # Hx1o vel, Hx0e vel, 2x1o boundary, 9x0e type
            edge_features_irreps=Irreps("1x1o + 1x0e"),  # 1o displacement, 0e distance
            lmax_attributes=args.config.lmax_attributes,
            n_vels=(args.config.input_seq_length - 1),
            velocity_aggregate=args.config.velocity_aggregate,
            attribute_mode=args.config.attribute_mode,
            homogeneous_particles=args.info.homogeneous_particles,
        )

    # transform core simulator outputting accelerations.
    model = hk.without_apply_rng(hk.transform_with_state(model))

    graph_tuple = graph_preprocess(features, particle_type[0])

    # mixed precision training based on this reference:
    # https://github.com/deepmind/dm-haiku/blob/main/examples/imagenet/train.py
    policy = jmp.get_policy("params=float32,compute=float32,output=float32")
    hk.mixed_precision.set_policy(MODEL, policy)

    params, state = model.init(subkey, graph_tuple)

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

    if args.config.mode == "train":
        train(
            model,
            params,
            state,
            neighbors,
            loader_train,
            loader_valid,
            setup,
            graph_preprocess,
            args,
        )
    elif args.config.mode == "infer":
        infer(
            model,
            params,
            state,
            neighbors,
            loader_valid,
            setup,
            graph_preprocess,
            args,
        )
