"""Source: https://github.com/wu375/simple-physics-simulator-pytorch-geometry"""

import functools
import json
import os
import pickle
import time

import numpy as np
import pyvista
import reading_utils
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
import torch
import tree
from torch import nn
from torch_geometric.nn import MessagePassing, radius_graph

import wandb

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# from tfrecord.torch.dataset import TFRecordDataset


dataset_name = "WaterDropSample"
DATA_DIR = os.path.join("GNS/data", dataset_name)
CKP_DIR = "GNS/data/ckp"
ROLLOUT_DIR = "GNS/data/rollouts"
os.makedirs(CKP_DIR, exist_ok=True)
os.makedirs(ROLLOUT_DIR, exist_ok=True)

is_train = True
if is_train:
    model_path = None
else:
    # model_path = res_dir + 'train_log/run0/model_973000.pth'
    model_path = os.path.join(CKP_DIR, "gns_WaterDropSample_1/model.pth")
is_write_vtk = False
is_wandb = False

particle_dimension = 3 if "3D" in dataset_name else 2
node_in = 37 if "3D" in dataset_name else 30
edge_in = 4 if "3D" in dataset_name else 3
batch_size = 1 if "3D" in dataset_name else 2

INPUT_SEQUENCE_LENGTH = 6
noise_std = 6.7e-4
training_steps = int(2e7)
log_steps = 5
eval_steps = 1000
save_steps = 1000

ARGS = {
    "is_train": is_train,
    "is_write_vtk": is_write_vtk,
    "dataset_name": dataset_name,
    "model_path": model_path,
    "input_sequence_length": INPUT_SEQUENCE_LENGTH,
    "bs": batch_size,
    "noise_std": noise_std,
    "training_steps": training_steps,
    "log_steps": training_steps,
    "eval_steps": eval_steps,
    "save_steps": save_steps,
}

device = "cuda"
with open(os.path.join(DATA_DIR, "metadata.json"), "rt") as f:
    metadata = json.loads(f.read())
num_steps = metadata["sequence_length"] - INPUT_SEQUENCE_LENGTH
normalization_stats = {
    "acceleration": {
        "mean": torch.FloatTensor(metadata["acc_mean"]).to(device),
        "std": torch.sqrt(
            torch.FloatTensor(metadata["acc_std"]) ** 2 + noise_std**2
        ).to(device),
    },
    "velocity": {
        "mean": torch.FloatTensor(metadata["vel_mean"]).to(device),
        "std": torch.sqrt(
            torch.FloatTensor(metadata["vel_std"]) ** 2 + noise_std**2
        ).to(device),
    },
}


def build_mlp(
    input_size,
    layer_sizes,
    output_size=None,
    output_activation=torch.nn.Identity,
    activation=torch.nn.ReLU,
):
    sizes = [input_size] + layer_sizes
    if output_size:
        sizes.append(output_size)

    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]
    return torch.nn.Sequential(*layers)


def time_diff(input_sequence):
    return input_sequence[:, 1:] - input_sequence[:, :-1]


def get_random_walk_noise_for_position_sequence(position_sequence, noise_std_last_step):
    """Returns random-walk noise in the velocity applied to the position."""
    velocity_sequence = time_diff(position_sequence)
    num_velocities = velocity_sequence.shape[1]
    velocity_sequence_noise = torch.randn(list(velocity_sequence.shape)) * (
        noise_std_last_step / num_velocities**0.5
    )

    velocity_sequence_noise = torch.cumsum(velocity_sequence_noise, dim=1)

    position_sequence_noise = torch.cat(
        [
            torch.zeros_like(velocity_sequence_noise[:, 0:1]),
            torch.cumsum(velocity_sequence_noise, dim=1),
        ],
        dim=1,
    )

    return position_sequence_noise


def _read_metadata(data_path):
    with open(os.path.join(data_path, "metadata.json"), "rt") as fp:
        return json.loads(fp.read())


class Encoder(nn.Module):
    """Encoder"""

    def __init__(
        self,
        node_in,
        node_out,
        edge_in,
        edge_out,
        mlp_num_layers,
        mlp_hidden_dim,
    ):
        super().__init__()
        self.node_fn = nn.Sequential(
            *[
                build_mlp(
                    node_in, [mlp_hidden_dim for _ in range(mlp_num_layers)], node_out
                ),
                nn.LayerNorm(node_out),
            ]
        )
        self.edge_fn = nn.Sequential(
            *[
                build_mlp(
                    edge_in, [mlp_hidden_dim for _ in range(mlp_num_layers)], edge_out
                ),
                nn.LayerNorm(edge_out),
            ]
        )

    def forward(self, x, _, e_features):  # edge_index, global_features
        # x: (E, node_in)
        # edge_index: (2, E)
        # e_features: (E, edge_in)
        return self.node_fn(x), self.edge_fn(e_features)


class InteractionNetwork(MessagePassing):
    """Interaction Network"""

    def __init__(
        self,
        node_in,
        node_out,
        edge_in,
        edge_out,
        mlp_num_layers,
        mlp_hidden_dim,
    ):
        super().__init__(aggr="add")
        self.node_fn = nn.Sequential(
            *[
                build_mlp(
                    node_in + edge_out,
                    [mlp_hidden_dim for _ in range(mlp_num_layers)],
                    node_out,
                ),
                nn.LayerNorm(node_out),
            ]
        )
        self.edge_fn = nn.Sequential(
            *[
                build_mlp(
                    node_in + node_in + edge_in,
                    [mlp_hidden_dim for _ in range(mlp_num_layers)],
                    edge_out,
                ),
                nn.LayerNorm(edge_out),
            ]
        )

    def forward(self, x, edge_index, e_features):
        # x: (E, node_in)
        # edge_index: (2, E)
        # e_features: (E, edge_in)
        x_residual = x
        e_features_residual = e_features
        x, e_features = self.propagate(
            edge_index=edge_index, x=x, e_features=e_features
        )
        return x + x_residual, e_features + e_features_residual

    def message(self, _, x_i, x_j, e_features):  # edge_index
        e_features = torch.cat([x_i, x_j, e_features], dim=-1)
        e_features = self.edge_fn(e_features)
        return e_features

    def update(self, x_updated, x, e_features):
        # x_updated: (E, edge_out)
        # x: (E, node_in)
        x_updated = torch.cat([x_updated, x], dim=-1)
        x_updated = self.node_fn(x_updated)
        return x_updated, e_features


class Processor(MessagePassing):
    """Processor"""

    def __init__(
        self,
        node_in,
        node_out,
        edge_in,
        edge_out,
        num_message_passing_steps,
        mlp_num_layers,
        mlp_hidden_dim,
    ):
        super().__init__(aggr="max")
        self.gnn_stacks = nn.ModuleList(
            [
                InteractionNetwork(
                    node_in=node_in,
                    node_out=node_out,
                    edge_in=edge_in,
                    edge_out=edge_out,
                    mlp_num_layers=mlp_num_layers,
                    mlp_hidden_dim=mlp_hidden_dim,
                )
                for _ in range(num_message_passing_steps)
            ]
        )

    def forward(self, x, edge_index, e_features):
        for gnn in self.gnn_stacks:
            x, e_features = gnn(x, edge_index, e_features)
        return x, e_features


class Decoder(nn.Module):
    """Decoder"""

    def __init__(
        self,
        node_in,
        node_out,
        mlp_num_layers,
        mlp_hidden_dim,
    ):
        super().__init__()
        self.node_fn = build_mlp(
            node_in, [mlp_hidden_dim for _ in range(mlp_num_layers)], node_out
        )

    def forward(self, x):
        # x: (E, node_in)
        return self.node_fn(x)


class EncodeProcessDecode(nn.Module):
    """Encode-Process-Decode model"""

    def __init__(
        self,
        node_in,
        node_out,
        edge_in,
        latent_dim,
        num_message_passing_steps,
        mlp_num_layers,
        mlp_hidden_dim,
    ):
        super().__init__()
        self._encoder = Encoder(
            node_in=node_in,
            node_out=latent_dim,
            edge_in=edge_in,
            edge_out=latent_dim,
            mlp_num_layers=mlp_num_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )
        self._processor = Processor(
            node_in=latent_dim,
            node_out=latent_dim,
            edge_in=latent_dim,
            edge_out=latent_dim,
            num_message_passing_steps=num_message_passing_steps,
            mlp_num_layers=mlp_num_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )
        self._decoder = Decoder(
            node_in=latent_dim,
            node_out=node_out,
            mlp_num_layers=mlp_num_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )

    def forward(self, x, edge_index, e_features):
        # x: (E, node_in)
        x, e_features = self._encoder(x, edge_index, e_features)
        x, e_features = self._processor(x, edge_index, e_features)
        x = self._decoder(x)
        return x


class Simulator(nn.Module):
    """Simulator"""

    def __init__(
        self,
        particle_dimension,
        node_in,
        edge_in,
        latent_dim,
        num_message_passing_steps,
        mlp_num_layers,
        mlp_hidden_dim,
        connectivity_radius,
        boundaries,
        normalization_stats,
        num_particle_types,
        particle_type_embedding_size,
        device="cuda",
    ):
        super().__init__()
        self._boundaries = boundaries
        self._connectivity_radius = connectivity_radius
        self._normalization_stats = normalization_stats
        self._num_particle_types = num_particle_types

        self._particle_type_embedding = nn.Embedding(
            num_particle_types, particle_type_embedding_size
        )  # (9, 16)

        self._encode_process_decode = EncodeProcessDecode(
            node_in=node_in,
            node_out=particle_dimension,
            edge_in=edge_in,
            latent_dim=latent_dim,
            num_message_passing_steps=num_message_passing_steps,
            mlp_num_layers=mlp_num_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )

        self._device = device

    def forward(self):
        pass

    def _build_graph_from_raw(
        self, position_sequence, n_particles_per_example, particle_types
    ):
        n_total_points = position_sequence.shape[0]
        most_recent_position = position_sequence[:, -1]  # (n_nodes, 2)
        velocity_sequence = time_diff(position_sequence)
        # senders and receivers are integers of shape (E,)
        senders, receivers = self._compute_connectivity(
            most_recent_position, n_particles_per_example, self._connectivity_radius
        )
        node_features = []
        # Normalized velocity sequence, merging spatial an time axis.
        velocity_stats = self._normalization_stats["velocity"]
        normalized_velocity_sequence = (
            velocity_sequence - velocity_stats["mean"]
        ) / velocity_stats["std"]
        flat_velocity_sequence = normalized_velocity_sequence.view(n_total_points, -1)
        node_features.append(flat_velocity_sequence)

        # Normalized clipped distances to lower and upper boundaries.
        # boundaries are an array of shape [num_dimensions, 2], where the second
        # axis, provides the lower/upper boundaries.
        boundaries = (
            torch.tensor(self._boundaries, requires_grad=False).float().to(self._device)
        )
        distance_to_lower_boundary = most_recent_position - boundaries[:, 0][None]
        distance_to_upper_boundary = boundaries[:, 1][None] - most_recent_position
        distance_to_boundaries = torch.cat(
            [distance_to_lower_boundary, distance_to_upper_boundary], dim=1
        )
        normalized_clipped_distance_to_boundaries = torch.clamp(
            distance_to_boundaries / self._connectivity_radius, -1.0, 1.0
        )
        node_features.append(normalized_clipped_distance_to_boundaries)

        if self._num_particle_types > 1:
            particle_type_embeddings = self._particle_type_embedding(particle_types)
            node_features.append(particle_type_embeddings)

        # Collect edge features.
        edge_features = []

        # Relative displacement and distances normalized to radius
        # (E, 2)
        # normalized_relative_displacements = (
        #     torch.gather(most_recent_position, 0, senders) - torch.gather(
        #         most_recent_position, 0, receivers)
        # ) / self._connectivity_radius
        normalized_relative_displacements = (
            most_recent_position[senders, :] - most_recent_position[receivers, :]
        ) / self._connectivity_radius
        edge_features.append(normalized_relative_displacements)

        normalized_relative_distances = torch.norm(
            normalized_relative_displacements, dim=-1, keepdim=True
        )
        edge_features.append(normalized_relative_distances)

        return (
            torch.cat(node_features, dim=-1),
            torch.stack([senders, receivers]),
            torch.cat(edge_features, dim=-1),
        )

    def _compute_connectivity(
        self, node_features, n_particles_per_example, radius, add_self_edges=True
    ):
        # handle batches. Default is 2 examples per batch

        # Specify examples id for particles/points
        batch_ids = torch.cat(
            [
                torch.LongTensor([i for _ in range(n)])
                for i, n in enumerate(n_particles_per_example)
            ]
        ).to(self._device)
        # radius = radius + 0.00001 # radius_graph takes r < radius not r <= radius
        edge_index = radius_graph(
            node_features, r=radius, batch=batch_ids, loop=add_self_edges
        )  # (2, n_edges)
        receivers = edge_index[0, :]
        senders = edge_index[1, :]
        return receivers, senders

    def _decoder_postprocessor(self, normalized_acceleration, position_sequence):
        # The model produces the output in normalized space so we apply inverse
        # normalization.
        acceleration_stats = self._normalization_stats["acceleration"]
        acceleration = (
            normalized_acceleration * acceleration_stats["std"]
        ) + acceleration_stats["mean"]

        # Use an Euler integrator to go from acceleration to position, assuming
        # a dt=1 corresponding to the size of the finite difference.
        most_recent_position = position_sequence[:, -1]
        most_recent_velocity = most_recent_position - position_sequence[:, -2]

        new_velocity = most_recent_velocity + acceleration  # * dt = 1
        new_position = most_recent_position + new_velocity  # * dt = 1
        return new_position

    def predict_positions(
        self, current_positions, n_particles_per_example, particle_types
    ):
        node_features, edge_index, e_features = self._build_graph_from_raw(
            current_positions, n_particles_per_example, particle_types
        )
        predicted_normalized_acceleration = self._encode_process_decode(
            node_features, edge_index, e_features
        )
        next_position = self._decoder_postprocessor(
            predicted_normalized_acceleration, current_positions
        )
        return next_position

    def predict_accelerations(
        self,
        next_position,
        position_sequence_noise,
        position_sequence,
        n_particles_per_example,
        particle_types,
    ):
        noisy_position_sequence = position_sequence + position_sequence_noise
        node_features, edge_index, e_features = self._build_graph_from_raw(
            noisy_position_sequence, n_particles_per_example, particle_types
        )
        predicted_normalized_acceleration = self._encode_process_decode(
            node_features, edge_index, e_features
        )
        next_position_adjusted = next_position + position_sequence_noise[:, -1]
        target_normalized_acceleration = self._inverse_decoder_postprocessor(
            next_position_adjusted, noisy_position_sequence
        )
        return predicted_normalized_acceleration, target_normalized_acceleration

    def _inverse_decoder_postprocessor(self, next_position, position_sequence):
        """Inverse of `_decoder_postprocessor`."""
        previous_position = position_sequence[:, -1]
        previous_velocity = previous_position - position_sequence[:, -2]
        next_velocity = next_position - previous_position
        acceleration = next_velocity - previous_velocity

        acceleration_stats = self._normalization_stats["acceleration"]
        normalized_acceleration = (
            acceleration - acceleration_stats["mean"]
        ) / acceleration_stats["std"]
        return normalized_acceleration

    def save(self, path="model.pth"):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


def write_vtk_temp(data_dict, path):
    """Store a .vtk file for ParaView"""
    r = np.asarray(data_dict["r"])
    N, dim = r.shape

    # PyVista treats the position information differently than the rest
    if dim == 2:
        r = np.hstack([r, np.zeros((N, 1))])
    data_pv = pyvista.PolyData(r)

    # copy all the other information also to pyvista, using plain numpy arrays
    for k, v in data_dict.items():
        # skip r because we already considered it above
        if k == "r":
            continue

        # working in 3D or scalar features do not require special care
        if dim == 2 and v.ndim == 2:
            v = np.hstack([v, np.zeros((N, 1))])

        data_pv[k] = np.asarray(v)

    data_pv.save(path)


def prepare_data_from_tfds(
    data_path=DATA_DIR + "/train.tfrecord", is_rollout=False, batch_size=2
):
    def prepare_inputs(tensor_dict):
        pos = tensor_dict["position"]
        pos = tf.transpose(pos, perm=[1, 0, 2])
        target_position = pos[:, -1]
        tensor_dict["position"] = pos[:, :-1]
        num_particles = tf.shape(pos)[0]
        tensor_dict["n_particles_per_example"] = num_particles[tf.newaxis]
        if "step_context" in tensor_dict:
            tensor_dict["step_context"] = tensor_dict["step_context"][-2]
            tensor_dict["step_context"] = tensor_dict["step_context"][tf.newaxis]
        return tensor_dict, target_position

    def batch_concat(dataset, batch_size):
        windowed_ds = dataset.window(batch_size)
        initial_state = tree.map_structure(
            lambda spec: tf.zeros(
                shape=[0] + spec.shape.as_list()[1:], dtype=spec.dtype
            ),
            dataset.element_spec,
        )

        def reduce_window(initial_state, ds):
            return ds.reduce(initial_state, lambda x, y: tf.concat([x, y], axis=0))

        return windowed_ds.map(
            lambda *x: tree.map_structure(reduce_window, initial_state, x)
        )

    def prepare_rollout_inputs(context, features):
        out_dict = {**context}
        pos = tf.transpose(features["position"], [1, 0, 2])
        target_position = pos[:, -1]
        out_dict["position"] = pos[:, :-1]
        out_dict["n_particles_per_example"] = [tf.shape(pos)[0]]
        if "step_context" in features:
            out_dict["step_context"] = features["step_context"]
        out_dict["is_trajectory"] = tf.constant([True], tf.bool)
        return out_dict, target_position

    metadata = _read_metadata(DATA_DIR)
    ds = tf.data.TFRecordDataset([data_path])
    is_one_traj = False
    if is_one_traj:
        ds = ds.take(1)
    ds = ds.map(
        functools.partial(
            reading_utils.parse_serialized_simulation_example, metadata=metadata
        )
    )
    if is_rollout:
        ds = ds.map(prepare_rollout_inputs)
    else:
        split_with_window = functools.partial(
            reading_utils.split_trajectory, window_length=6 + 1
        )
        ds = ds.flat_map(split_with_window)
        ds = ds.map(prepare_inputs)
        # ds = ds.repeat() # repeats the dataset indefinitely
        # ds = ds.shuffle(512)
        ds = batch_concat(ds, batch_size)
    ds = tfds.as_numpy(ds)
    for _ in range(5):  # clear screen
        print()
    return ds


def eval_single_rollout(simulator, features, num_steps, device):
    initial_positions = features["position"][
        :, 0:INPUT_SEQUENCE_LENGTH
    ]  # (n_nodes, t_window, dim)
    ground_truth_positions = features["position"][
        :, INPUT_SEQUENCE_LENGTH:
    ]  # (n_nodes, traj_len - t_window, dim)

    current_positions = initial_positions  # (n_nodes, t_window, dim)
    predictions = torch.zeros_like(ground_truth_positions).permute(
        1, 0, 2
    )  # (traj_len - t_window, n_nodes, dim)
    for step in range(num_steps):
        next_position = simulator.predict_positions(
            current_positions,
            n_particles_per_example=features["n_particles_per_example"],
            particle_types=features["particle_type"],
        )  # (n_nodes, dim)
        # Update kinematic particles from prescribed trajectory.
        kinematic_mask = (features["particle_type"] == 3).clone().detach().to(device)
        next_position_ground_truth = ground_truth_positions[:, step]
        kinematic_mask = kinematic_mask.bool()[:, None].expand(-1, particle_dimension)
        next_position = torch.where(
            kinematic_mask, next_position_ground_truth, next_position
        )
        predictions[step] = next_position
        current_positions = torch.cat(
            [current_positions[:, 1:], next_position[:, None, :]], dim=1
        )

    ground_truth_positions = ground_truth_positions.permute(
        1, 0, 2
    )  # (traj_len - t_window, n_nodes, dim)
    loss = ((predictions - ground_truth_positions) ** 2).mean()
    output_dict = {
        "initial_positions": initial_positions.permute(1, 0, 2).cpu().numpy(),
        "predicted_rollout": predictions.cpu().numpy(),
        "ground_truth_rollout": ground_truth_positions.cpu().numpy(),
        "particle_types": features["particle_type"].cpu().numpy(),
    }
    return output_dict, loss


def eval_rollout(
    simulator, num_steps, num_eval_steps=1, save_results=False, device="cuda"
):
    ds = prepare_data_from_tfds(
        data_path=os.path.join(DATA_DIR, "valid.tfrecord"), is_rollout=True
    )

    eval_loss = []
    i = 0
    simulator.eval()
    with torch.no_grad():
        for example_i, (features, labels) in enumerate(ds):
            features["position"] = torch.tensor(features["position"]).to(
                device
            )  # (n_nodes, 1000, 2)
            features["n_particles_per_example"] = torch.tensor(
                features["n_particles_per_example"]
            ).to(device)
            features["particle_type"] = torch.tensor(features["particle_type"]).to(
                device
            )
            labels = torch.tensor(labels).to(
                device
            )  # TODO: not used. Final state 1001 of the trajectory. (n_nodes, 2)
            example_rollout, loss = eval_single_rollout(
                simulator, features, num_steps, device
            )
            eval_loss.append(loss)
            if save_results and (model_path is not None):
                example_rollout["metadata"] = metadata
                run_name = model_path.split("/")[-2]
                rollout_dir = os.path.join(ROLLOUT_DIR, run_name)
                os.makedirs(rollout_dir, exist_ok=True)

                file_prefix = f"{rollout_dir}/rollout_{example_i}"
                if is_write_vtk:

                    for j in range(example_rollout["predicted_rollout"].shape[0]):
                        filename_vtk = file_prefix + f"_{j}.vtk"
                        state = {"r": example_rollout["predicted_rollout"][j]}
                        write_vtk_temp(state, filename_vtk)

                    for j in range(example_rollout["ground_truth_rollout"].shape[0]):
                        filename_vtk = file_prefix + f"_ref_{j}.vtk"
                        state = {"r": example_rollout["ground_truth_rollout"][j]}
                        write_vtk_temp(state, filename_vtk)
                else:
                    filename = f"{file_prefix}.pkl"

                    with open(filename, "wb") as f:
                        pickle.dump(example_rollout, f)
            i += 1
            if i >= num_eval_steps:
                break
    simulator.train()
    return torch.stack(eval_loss).mean().item()


def train(simulator):
    run_prefix = "_".join(["gns", dataset_name, ""])

    i = 0
    while os.path.isdir(os.path.join(CKP_DIR, run_prefix + str(i))):
        i += 1
    run_name = run_prefix + str(i)
    ckp_dir = os.path.join(CKP_DIR, run_name)
    os.makedirs(ckp_dir, exist_ok=True)

    if is_wandb:
        wandb.init(
            project="segnn",
            entity="segnn-sph",
            name=run_name,
            config=ARGS,
            save_code=True,
        )

    lr_init = 1e-4
    # lr_min = 1e-6
    lr_decay = 0.1
    lr_decay_steps = int(5e6)
    lr_new = lr_init
    optimizer = torch.optim.Adam(simulator.parameters(), lr=lr_init)

    step = 0
    try:

        for _ in range(5):
            ds = prepare_data_from_tfds(
                DATA_DIR + "/train.tfrecord", batch_size=batch_size
            )
            sample = next(iter(ds))
            print(sample[0]["position"].shape)

            start = time.time()
            step = 0
            for features, labels in ds:
                # for _ in range(1001):
                #     features, labels = _features, _labels
                features["position"] = torch.tensor(features["position"]).to(device)
                features["n_particles_per_example"] = torch.tensor(
                    features["n_particles_per_example"]
                ).to(device)
                features["particle_type"] = torch.tensor(features["particle_type"]).to(
                    device
                )
                labels = torch.tensor(labels).to(device)

                sampled_noise = get_random_walk_noise_for_position_sequence(
                    features["position"], noise_std_last_step=noise_std
                ).to(device)
                non_kinematic_mask = (
                    (features["particle_type"] != 3).clone().detach().to(device)
                )
                sampled_noise *= non_kinematic_mask.view(-1, 1, 1)

                pred, target = simulator.predict_accelerations(
                    next_position=labels,
                    position_sequence_noise=sampled_noise,
                    position_sequence=features["position"],
                    n_particles_per_example=features["n_particles_per_example"],
                    particle_types=features["particle_type"],
                )
                loss = (pred - target) ** 2
                loss = loss.sum(dim=-1)
                num_non_kinematic = non_kinematic_mask.sum()

                loss = torch.where(
                    non_kinematic_mask.bool(), loss, torch.zeros_like(loss)
                )
                loss = loss.sum() / num_non_kinematic

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(step)
                step += 1
                if step % log_steps == 0 and is_wandb:
                    wandb.log({"train/loss": loss.item(), "lr": lr_new}, step)

                lr_new = lr_init * (lr_decay ** (step / lr_decay_steps))
                for g in optimizer.param_groups:
                    g["lr"] = lr_new

                step += 1
                print(
                    f"Training step: {step}/{training_steps}. Loss: {loss.item()}."
                )  # , end="\r",)
                if step >= training_steps == 0:
                    break

                if step % eval_steps == 0 and is_wandb:
                    eval_loss = eval_rollout(
                        simulator, num_steps, num_eval_steps=5, device=device
                    )
                    wandb.log({"val/loss": eval_loss}, step)

                if step % save_steps == 0:
                    simulator.save(os.path.join(ckp_dir, "model.pth"))

            print(time.time() - start)

    except KeyboardInterrupt:
        print(features["position"].shape)
        pass

    simulator.save(os.path.join(ckp_dir, "model.pth"))


def infer(simulator):
    _ = eval_rollout(simulator, num_steps=num_steps, save_results=True, device=device)


if __name__ == "__main__":
    simulator = Simulator(
        particle_dimension=particle_dimension,
        node_in=node_in,
        edge_in=edge_in,
        latent_dim=128,
        num_message_passing_steps=2,
        mlp_num_layers=10,
        mlp_hidden_dim=128,
        connectivity_radius=metadata["default_connectivity_radius"],
        boundaries=np.array(metadata["bounds"]),
        normalization_stats=normalization_stats,
        num_particle_types=9,
        particle_type_embedding_size=16,
        device=device,
    )
    if model_path is not None:
        simulator.load(model_path)
    if device == "cuda":
        simulator.cuda()

    if is_train:
        train(simulator)
    else:
        infer(simulator)
