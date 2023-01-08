import argparse
import functools
import json
import os

import h5py
import numpy as np
import reading_utils
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""


def extract_one_trajectory_from_tfrecord_and_write_to_tfrecord(args):
    """
    Read a tfrecord file, extract one trajectory, and write it to a new
    tfrecord file.
    """
    # TODO: remove hard-coded values
    args.dataset_path = "GNS/data/WaterDropSample"
    args.file_name = "train.tfrecord"

    src_path = os.path.join(args.dataset_path, args.file_name)

    with open(os.path.join(args.dataset_path, "metadata.json"), "r") as fp:
        metadata = json.loads(fp.read())

    # get the TFRecordDataset with its proper preprocessing
    src = tf.data.TFRecordDataset([src_path])
    src = src.map(
        functools.partial(
            reading_utils.parse_serialized_simulation_example, metadata=metadata
        )
    )
    src = tfds.as_numpy(src)

    # destination dataset
    dst_name = args.file_name[:-9] + "_one_traj.tfrecord"
    dst_path = os.path.join(args.dataset_path, dst_name)

    with tf.io.TFRecordWriter(dst_path) as writer:
        for i, elem in enumerate(src):
            if i == 0:
                writer.write(elem)
                break


def convert_tfrecord_to_h5(args):
    """Read .tfrecord file and convert it to its closest .h5 equivalent"""

    file_path = os.path.join(args.dataset_path, args.file_name)

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

    hf = h5py.File(f"{file_path[:-9]}.h5", "w")

    for i, elem in enumerate(ds):
        traj_str = str(i).zfill(4)

        particle_type = elem[0]["particle_type"]
        key = elem[0]["key"]
        position = elem[1]["position"]

        hf.create_dataset(f"{traj_str}/particle_type", data=particle_type)
        assert key == i, "Something is wrong here"
        # hf.create_dataset(f'{traj_str}/key', data=key)
        hf.create_dataset(
            f"{traj_str}/position",
            data=position,
            dtype=np.float32,
            compression=args.compression,
        )

    hf.close()


def read_h5_demo(args):
    """Simple demonstation how to read the previously converted h5 file"""

    file_path = os.path.join(args.dataset_path, args.file_name)
    hf = h5py.File(f"{file_path[:-9]}.h5", "r")

    for key in hf.keys():
        r = hf[f"{key}/r"][:]
        print(key, r.shape, r.mean())

    hf.close()


def compute_statistics_tfrecord(args):
    """Compute the mean and std of a TFRecordDataset"""

    with open(os.path.join(args.dataset_path, "metadata.json"), "r") as fp:
        metadata = json.loads(fp.read())

    vels, accs = [], []
    vels_sq, accs_sq = [], []
    for loop in ["mean", "std"]:
        for split in ["train", "valid", "test"]:
            file_path = os.path.join(args.dataset_path, split + ".tfrecord")

            # get the TFRecordDataset with its proper preprocessing
            ds = tf.data.TFRecordDataset([file_path])
            ds = ds.map(
                functools.partial(
                    reading_utils.parse_serialized_simulation_example, metadata=metadata
                )
            )
            ds = tfds.as_numpy(ds)

            for elem in ds:
                r = elem[1]["position"]

                if loop == "mean":
                    vels.append((r[1:] - r[:-1]).mean((0, 1)))
                    accs.append((r[2:] + r[:-2] - 2 * r[1:-1]).mean((0, 1)))
                elif loop == "std":
                    centered_vel = r[1:] - r[:-1] - vel_mean
                    vels_sq.append(np.square(centered_vel).mean((0, 1)))
                    centered_acc = r[2:] + r[:-2] - 2 * r[1:-1] - acc_mean
                    accs_sq.append(np.square(centered_acc).mean((0, 1)))

        if loop == "mean":
            vel_mean = np.stack(vels).mean(0)
            acc_mean = np.stack(accs).mean(0)
            print(f"vel_mean={vel_mean}, acc_mean={acc_mean}")
        elif loop == "std":
            vel_std = np.stack(vels_sq).mean(0) ** 0.5
            acc_std = np.stack(accs_sq).mean(0) ** 0.5
            print(f"vel_std={vel_std}, acc_std={acc_std}")


if __name__ == "__main__":

    # Environment setup:

    # python3 -m venv venv_tf
    # source venv_tf/bin/activate
    # pip install tensorflow tensorflow-datasets

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="GNS/data/WaterDropSample",
        help="Location of the dataset",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="train.tfrecord",
        help="Which file to convert from .tfrecord to .h5",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="None",
        choices=["None", "gzip"],
        help='"gzip" takes 8x longer, but reduces size by 15%',
    )
    args = parser.parse_args()
    args.compression = "gzip" if args.compression == "gzip" else None

    # convert_tfrecord_to_h5(args)
    # compute_statistics_tfrecord(args)
    # extract_one_trajectory_from_tfrecord_and_write_to_tfrecord(args)
