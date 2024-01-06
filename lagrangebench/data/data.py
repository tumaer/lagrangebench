"""Dataset modules for loading HDF5 simulation trajectories."""

import bisect
import json
import os
import os.path as osp
import re
import zipfile
from typing import Callable, Optional

import h5py
import jax.numpy as jnp
import numpy as np
import wget
from torch.utils.data import Dataset

from lagrangebench.utils import NodeType

URLS = {
    "tgv2d": "https://zenodo.org/records/10021926/files/2D_TGV_2500_10kevery100.zip",
    "rpf2d": "https://zenodo.org/records/10021926/files/2D_RPF_3200_20kevery100.zip",
    "ldc2d": "https://zenodo.org/records/10021926/files/2D_LDC_2708_10kevery100.zip",
    "dam2d": "https://zenodo.org/records/10021926/files/2D_DAM_5740_20kevery100.zip",
    "tgv3d": "https://zenodo.org/records/10021926/files/3D_TGV_8000_10kevery100.zip",
    "rpf3d": "https://zenodo.org/records/10021926/files/3D_RPF_8000_10kevery100.zip",
    "ldc3d": "https://zenodo.org/records/10021926/files/3D_LDC_8160_10kevery100.zip",
}


def get_dataset_name_from_path(path: str) -> str:
    """Infer the dataset name from the provided path.

    This function assumes that the dataset directory name has the following structure:
    {2D|3D}_{TGV|RPF|LDC|DAM}_{num_particles_max}_{num_steps}every{sampling_rate}

    The dataset name then becomes one of the following:
    {tgv2d|tgv3d|rpf2d|rpf3d|ldc2d|ldc3d|dam2d}
    """

    name = re.search(r"(?:2D|3D)_[A-Z]{3}", path)
    assert name is not None, (
        f"No valid dataset name found in path {path}. "
        "Valid name formats: {2D|3D}_{TGV|RPF|LDC|DAM} "
        "Alternatively, you can specify the dataset name explicitly."
    )
    name = name.group(0)
    name = f"{name.split('_')[1]}{name.split('_')[0]}".lower()
    return name


class H5Dataset(Dataset):
    """Dataset for loading HDF5 simulation trajectories.

    Reference on parallel loading of h5 samples see:
    https://github.com/pytorch/pytorch/issues/11929

    Implementation inspired by:
    https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/datasets/lmdb_dataset.py
    """

    def __init__(
        self,
        split: str,
        dataset_path: str,
        name: Optional[str] = None,
        input_seq_length: int = 6,
        n_rollout_steps: int = 0,
        nl_backend: str = "jaxmd_vmap",
        external_force_fn: Optional[Callable] = None,
    ):
        """Initialize the dataset. If the dataset is not present, it is downloaded.

        Args:
            split: "train", "valid", or "test"
            dataset_path: Path to the dataset
            name: Name of the dataset. If None, it is inferred from the path.
            input_seq_length: Length of the input sequence. The number of historic
                velocities is input_seq_length - 1. And during training, the returned
                number of past positions is input_seq_length + 1, to compute target
                acceleration.
            n_rollout_steps: During training, this is the maximum number of pushforward
                unroll steps. During validation/testing, this specifies the largest
                N-step MSE loss we are interested in, e.g. for best model checkpointing.
            nl_backend: Which backend to use for the neighbor list
            external_force_fn: Function that returns the position-wise external force
        """

        if dataset_path.endswith("/"):  # remove trailing slash in dataset path
            dataset_path = dataset_path[:-1]

        if name is None:
            self.name = get_dataset_name_from_path(dataset_path)
        if not osp.exists(dataset_path): #download the dataset if it is not present
            dataset_path = self.download(self.name, dataset_path)

        assert split in ["train", "valid", "test"]
        assert (
            input_seq_length > 1
        ), "To compute at least one past velocity, input_seq_length must be >= 2."
        self.dataset_path = dataset_path
        self.file_path = osp.join(dataset_path, split + ".h5")
        self.input_seq_length = input_seq_length
        self.nl_backend = nl_backend

        self.external_force_fn = external_force_fn

        # load dataset metadata
        with open(osp.join(dataset_path, "metadata.json"), "r") as f:
            self.metadata = json.loads(f.read())

        self.db_hdf5 = None

        with h5py.File(self.file_path, "r") as f:
            self.traj_keys = list(f.keys())

            # (num_steps, num_particles, dim) = f["00000/position"].shape
            self.sequence_length = f["00000/position"].shape[0]   #20001 for RPF2D

        if split == "train":
            # During training, the first input_seq_length steps can only be used as
            # input, and the last one to compute the target acceleration. If we use
            # pushforward, then we need to provide n_rollout_steps more steps
            # from the end of a trajectory. Thus, the number of training samples per
            # trajectory becomes:
            self.subseq_length = input_seq_length + 1 + n_rollout_steps #7 (6+1+0) if there is no pushforward i.e. n_rollout_steps=0
            samples_per_traj = self.sequence_length - self.subseq_length + 1 #(20001-7+1=19995)

            keylens = jnp.array([samples_per_traj for _ in range(len(self.traj_keys))]) #Array([19995])
            self._keylen_cumulative = jnp.cumsum(keylens).tolist() #[19995]

            self.num_samples = sum(keylens) #Array(19995, dtype=int64)
            self.getter = self.get_window

        else:
            assert (
                n_rollout_steps > 0
            ), "n_rollout_steps must be > 0 for validation and testing."
            # Compute the number of splits per validation trajectory. If the length of
            # each trajectory is 1000, we want to compute a 20-step MSE, and
            # intput_seq_length=6, then we should split the trajectory into
            # _split_valid_traj_into_n = 1000 // (20 + 6) chunks.
            self.subseq_length = input_seq_length + n_rollout_steps
            self._split_valid_traj_into_n = self.sequence_length // self.subseq_length

            self.num_samples = self._split_valid_traj_into_n * len(self.traj_keys)
            self.getter = self.get_trajectory

        assert self.sequence_length >= self.subseq_length, (
            f"# steps in dataset trajectory ({self.sequence_length}) must be >= "
            f"subsequence length ({self.subseq_length}). Reduce either "
            f"input_seq_length or n_rollout_steps/max pushforward steps."
        )

    def download(self, name: str, path: str) -> str:
        """Download the dataset.

        Args:
            name: Name of the dataset
            path: Destination path to the downloaded dataset
        """

        assert name in URLS, f"Dataset {name} not available."
        url = URLS[name]

        # path could be e.g. "./data/2D_TGV_2500_10kevery100/"
        # remove trailing slash if present and get the root of the datasets
        path = path[:-1] if path.endswith("/") else path
        path_root = osp.split(path)[0]  # e.g. # "./data"

        # download the dataset as a zip file, e.g. "./data/2D_TGV_2500_10kevery100.zip"
        os.makedirs(path_root, exist_ok=True)
        filename = wget.download(url, out=path_root)
        print(f"Dataset {name} downloaded to {filename}")

        # unzip the dataset and then remove the zip file
        zipfile.ZipFile(filename, "r").extractall(path_root)
        os.remove(filename)

        return path

    def _open_hdf5(self) -> h5py.File:
        if self.db_hdf5 is None:
            return h5py.File(self.file_path, "r")
        else:
            return self.db_hdf5

    def _matscipy_pad(self, pos_input, particle_type):
        padding_size = self.metadata["num_particles_max"] - pos_input.shape[0]
        pos_input = np.pad(
            pos_input,
            ((0, padding_size), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )
        particle_type = np.pad(
            particle_type,
            (0, padding_size),
            mode="constant",
            constant_values=NodeType.PAD_VALUE,
        )
        return pos_input, particle_type

    def get_trajectory(self, idx: int):
        """Get a (full) trajectory and index idx."""
        # open the database file
        self.db_hdf5 = self._open_hdf5()

        if self._split_valid_traj_into_n > 1:
            traj_idx = idx // self._split_valid_traj_into_n
            slice_from = (idx % self._split_valid_traj_into_n) * self.subseq_length
            slice_to = slice_from + self.subseq_length
        else:
            traj_idx = idx
            slice_from = 0
            slice_to = self.sequence_length

        # get a pointer to the trajectory. That is not yet the real trajectory.
        traj = self.db_hdf5[f"{self.traj_keys[traj_idx]}"]
        # get a pointer to the positions of the traj. Still nothing in memory.
        traj_pos = traj["position"]
        # load and transpose the trajectory
        pos_input = traj_pos[slice_from:slice_to].transpose((1, 0, 2))

        particle_type = traj["particle_type"][:]

        if self.nl_backend == "matscipy":
            pos_input, particle_type = self._matscipy_pad(pos_input, particle_type)

        return pos_input, particle_type

    def get_window(self, idx: int):
        """Get a window of the trajectory and index idx."""
        # figure out which trajectory this should be indexed from.
        traj_idx = bisect.bisect(self._keylen_cumulative, idx)
        # extract index of element within that trajectory.
        el_idx = idx
        if traj_idx != 0:
            el_idx = idx - self._keylen_cumulative[traj_idx - 1]
        assert el_idx >= 0

        # open the database file
        self.db_hdf5 = self._open_hdf5()

        # get a pointer to the trajectory. That is not yet the real trajectory.
        traj = self.db_hdf5[f"{self.traj_keys[traj_idx]}"]
        # get a pointer to the positions of the traj. Still nothing in memory.
        traj_pos = traj["position"]
        # load only a slice of the positions. Now, this is an array in memory.
        pos_input_and_target = traj_pos[el_idx : el_idx + self.subseq_length]
        pos_input_and_target = pos_input_and_target.transpose((1, 0, 2))

        particle_type = traj["particle_type"][:]

        if self.nl_backend == "matscipy":
            pos_input_and_target, particle_type = self._matscipy_pad(
                pos_input_and_target, particle_type
            )

        return pos_input_and_target, particle_type

    def __getitem__(self, idx: int):
        return self.getter(idx)

    def __len__(self):
        return self.num_samples


class TGV2D(H5Dataset):
    """Taylor-Green Vortex 2D dataset. 2.5K particles."""

    def __init__(
        self,
        split: str,
        dataset_path: str = "datasets/2D_TGV_2500_10kevery100",
        input_seq_length: int = 6,
        n_rollout_steps: int = 0,
        nl_backend: str = "jaxmd_vmap",
    ):
        super().__init__(
            split,
            dataset_path,
            name="tgv2d",
            input_seq_length=input_seq_length,
            n_rollout_steps=n_rollout_steps,
            nl_backend=nl_backend,
        )


class TGV3D(H5Dataset):
    """Taylor-Green Vortex 3D dataset. 8K particles."""

    def __init__(
        self,
        split: str,
        dataset_path: str = "datasets/3D_TGV_8000_10kevery100",
        input_seq_length: int = 6,
        n_rollout_steps: int = 0,
        nl_backend: str = "jaxmd_vmap",
    ):
        super().__init__(
            split,
            dataset_path,
            name="tgv3d",
            input_seq_length=input_seq_length,
            n_rollout_steps=n_rollout_steps,
            nl_backend=nl_backend,
        )


class RPF2D(H5Dataset):
    """Reverse Poiseuille Flow 2D dataset. 3.2K particles."""

    def __init__(
        self,
        split: str,
        dataset_path: str = "datasets/2D_RPF_3200_20kevery100",
        input_seq_length: int = 6,
        n_rollout_steps: int = 0,
        nl_backend: str = "jaxmd_vmap",
    ):
        def external_force_fn(position):
            return jnp.where(
                position[1] > 1.0,
                jnp.array([-1.0, 0.0]),
                jnp.array([1.0, 0.0]),
            )

        super().__init__(
            split,
            dataset_path,
            name="rpf2d",
            input_seq_length=input_seq_length,
            n_rollout_steps=n_rollout_steps,
            nl_backend=nl_backend,
            external_force_fn=external_force_fn,
        )


class RPF3D(H5Dataset):
    """Reverse Poiseuille Flow 3D dataset. 8K particles."""

    def __init__(
        self,
        split: str,
        dataset_path: str = "datasets/3D_RPF_8000_10kevery100",
        input_seq_length: int = 6,
        n_rollout_steps: int = 0,
        nl_backend: str = "jaxmd_vmap",
    ):
        def external_force_fn(position):
            return jnp.where(
                position[1] > 1.0,
                jnp.array([-1.0, 0.0, 0.0]),
                jnp.array([1.0, 0.0, 0.0]),
            )

        super().__init__(
            split,
            dataset_path,
            name="rpf3d",
            input_seq_length=input_seq_length,
            n_rollout_steps=n_rollout_steps,
            nl_backend=nl_backend,
            external_force_fn=external_force_fn,
        )


class LDC2D(H5Dataset):
    """Lid-Driven Cabity 2D dataset. 2.5K particles."""

    def __init__(
        self,
        split: str,
        dataset_path: str = "datasets/2D_LDC_2500_10kevery100",
        input_seq_length: int = 6,
        n_rollout_steps: int = 0,
        nl_backend: str = "jaxmd_vmap",
    ):
        super().__init__(
            split,
            dataset_path,
            name="ldc2d",
            input_seq_length=input_seq_length,
            n_rollout_steps=n_rollout_steps,
            nl_backend=nl_backend,
        )


class LDC3D(H5Dataset):
    """Lid-Driven Cabity 3D dataset. 8.2K particles."""

    def __init__(
        self,
        split: str,
        dataset_path: str = "datasets/3D_LDC_8160_10kevery100",
        input_seq_length: int = 6,
        n_rollout_steps: int = 0,
        nl_backend: str = "jaxmd_vmap",
    ):
        super().__init__(
            split,
            dataset_path,
            name="ldc3d",
            input_seq_length=input_seq_length,
            n_rollout_steps=n_rollout_steps,
            nl_backend=nl_backend,
        )


class DAM2D(H5Dataset):
    """Dam break 2D dataset. 5.7K particles."""

    def __init__(
        self,
        split: str,
        dataset_path: str = "datasets/2D_DB_5740_20kevery100",
        input_seq_length: int = 6,
        n_rollout_steps: int = 0,
        nl_backend: str = "jaxmd_vmap",
    ):
        super().__init__(
            split,
            dataset_path,
            name="dam2d",
            input_seq_length=input_seq_length,
            n_rollout_steps=n_rollout_steps,
            nl_backend=nl_backend,
        )
