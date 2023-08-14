import argparse
import functools
import json
import os

import h5py
import numpy as np
import reading_utils
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def convert_tfrecord_to_h5(args):
    """Read .tfrecord file and convert it to its closest .h5 equivalent"""

    file_path = os.path.join(args.dataset_path, args.file_name)
    print(f"Start conversion of {file_path} to .h5")

    with open(os.path.join(args.dataset_path, "metadata.json"), "r") as fp:
        metadata = json.loads(fp.read())

    # get the TFRecordDataset with its proper preprocessing
    ds = tf.data.TFRecordDataset([file_path])
    ds = ds.map(
        functools.partial(
            reading_utils.parse_serialized_simulation_example, metadata=metadata
        )
    )
    ds = tfds.as_numpy(ds)

    h5_file_path = f"{file_path[:-9]}.h5"
    hf = h5py.File(h5_file_path, "w")

    for i, elem in enumerate(ds):
        traj_str = str(i).zfill(5)

        particle_type = elem[0]["particle_type"]
        key = elem[0]["key"]
        position = elem[1]["position"]

        hf.create_dataset(f"{traj_str}/particle_type", data=particle_type)
        assert key == i, "Something went wrong here"
        hf.create_dataset(
            f"{traj_str}/position",
            data=position,
            dtype=np.float32,
            compression="gzip",
        )

    hf.close()
    print(f"Finish conversion to {h5_file_path}")


def main(args):
    files = os.listdir(args.dataset_path)
    files = [f for f in files if f.endswith(".tfrecord")]
    for file_name in files:
        args.file_name = file_name
        convert_tfrecord_to_h5(args)

    # add the maximum number of particles to the metadata
    # Crucial for the matscipy neighbors search

    # first find the maximum number of particles
    files = os.listdir(args.dataset_path)
    files = [f for f in files if f.endswith(".h5")]
    max_particles = 0
    for file_name in files:
        h5_file_path = os.path.join(args.dataset_path, file_name)
        hf = h5py.File(h5_file_path, "r")
        for k, v in hf.items():
            max_particles = max(v["particle_type"].shape[0], max_particles)
        print(f"Max number of particles in {file_name}: {max_particles}")
        hf.close()

    metadata_path = os.path.join(args.dataset_path, "metadata.json")
    with open(metadata_path, "r") as fp:
        metadata = json.loads(fp.read())

    metadata["num_particles_max"] = max_particles
    # all DeepMind datasets are non-periodic
    metadata["periodic_boundary_conditions"] = [False, False, False]

    with open(metadata_path, "w") as f:
        json.dump(metadata, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str)
    args = parser.parse_args()

    main(args)
