import bisect
import os

import h5py
import numpy as np
from torch.utils.data import Dataset


class H5Dataset(Dataset):
    """Dataset for loading h5 simulation trajectories."""

    def __init__(
        self,
        dataset_path: str,
        split: str,
        input_seq_length: int = 6,
        split_valid_traj_into_n: int = 1,
        is_rollout: bool = False,
    ):
        """
        Reference on parallel loading of h5 samples see:
        https://github.com/pytorch/pytorch/issues/11929

        Implementation inspired by:
        https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/datasets/lmdb_dataset.py

        """

        self.dataset_path = dataset_path
        self.file_path = os.path.join(dataset_path, split + ".h5")
        self.input_seq_length = input_seq_length
        self.split_valid_traj_into_n = split_valid_traj_into_n

        with h5py.File(self.file_path, "r") as f:
            self.traj_keys = list(f.keys())

            self.sequence_length = f["0000/position"].shape[0]

            # # the true sequence length is not the one in the metadata file. The
            # # very first frame is needed to compute the velocity.
            # with open(os.path.join(dataset_path, "metadata.json"), "r") as fp:
            #     metadata = json.loads(fp.read())
            # sequence_length = metadata["sequence_length"] + 1

            # For some datasets the number of particles per trajectory varies.
            # For the purpose of batching we need to pad the trajectories to
            # the maximum number of particles. This is done in the dataloader.
            # The following lines are for debugging purposes only and were used
            # for experimenting without padding.
            # self.traj_num_nodes = [
            #     f[f"{k}/position"].shape[1] for k in f.keys()]
            # tmp = np.array(sorted(self.traj_num_nodes))
            # print(f"bs=2, worst case: {tmp[len(tmp)//2] + tmp[-1]} nodes")
            # print(f"Number of nodes min={tmp.min()}, max={tmp.max()}, "
            #       f"mean={tmp.mean()}")

        # Subsequence is used to split one long validation trajectory into multiple
        self.subsequence_length = self.sequence_length // split_valid_traj_into_n

        if is_rollout:
            self.getter = self.get_trajectory
            self.num_samples = split_valid_traj_into_n * len(self.traj_keys)
        else:
            samples_per_traj = self.sequence_length - self.input_seq_length - 1
            keylens = [samples_per_traj for _ in range(len(self.traj_keys))]
            self._keylen_cumulative = np.cumsum(keylens).tolist()
            self.num_samples = sum(keylens)

            self.getter = self.get_window

    def open_hdf5(self):
        self.db_hdf5 = h5py.File(self.file_path, "r")

    def get_trajectory(self, idx: int):
        # Open the database file
        if not hasattr(self, "db_hdf5"):
            self.open_hdf5()

        if self.split_valid_traj_into_n > 1:
            traj_idx = idx // self.split_valid_traj_into_n
            slice_from = (idx % self.split_valid_traj_into_n) * self.subsequence_length
            slice_to = slice_from + self.subsequence_length
        else:
            traj_idx = idx
            slice_from = 0
            slice_to = self.sequence_length

        # Get a pointer to the trajectory. That is not yet the real trajectory.
        traj = self.db_hdf5[f"{self.traj_keys[traj_idx]}"]
        # Get a pointer to the positions of the traj. Still nothing in memory.
        traj_pos = traj["position"]
        # load and transpose the trajectory
        pos_input = traj_pos[slice_from:slice_to].transpose((1, 0, 2))

        particle_type = traj["particle_type"][:]

        return pos_input, particle_type

    def get_window(self, idx: int):
        # Figure out which trajectory this should be indexed from.
        traj_idx = bisect.bisect(self._keylen_cumulative, idx)
        # Extract index of element within that trajectory.
        el_idx = idx
        if traj_idx != 0:
            el_idx = idx - self._keylen_cumulative[traj_idx - 1]
        assert el_idx >= 0

        # Open the database file
        if not hasattr(self, "db_hdf5"):
            self.open_hdf5()

        # Get a pointer to the trajectory. That is not yet the real trajectory.
        traj = self.db_hdf5[f"{self.traj_keys[traj_idx]}"]
        # Get a pointer to the positions of the traj. Still nothing in memory.
        traj_pos = traj["position"]
        # Load only a slice of the positions. Now, this is an array in memory.
        pos_input_and_target = traj_pos[el_idx : el_idx + self.input_seq_length + 1]
        pos_input_and_target = pos_input_and_target.transpose((1, 0, 2))

        particle_type = traj["particle_type"][:]

        return pos_input_and_target, particle_type

    def __getitem__(self, idx: int):
        return self.getter(idx)

    def __len__(self):
        return self.num_samples


def numpy_collate(batch):
    """
    Source:
    https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html
    """
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)  # TODO: JAX
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)  # TODO: JAX
