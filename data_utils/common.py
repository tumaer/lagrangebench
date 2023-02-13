import json
import os

import h5py
import numpy as np


def compute_statistics_h5(dataset_root="GNS/data/BoxBath"):
    """Compute the mean and std of a h5 dataset files"""

    vels, accs = [], []
    vels_sq, accs_sq = [], []
    for loop in ["mean", "std"]:
        for split in ["train", "valid"]:
            hf = h5py.File(f"{dataset_root}/{split}.h5", "r")

            for _, v in hf.items():
                r = v.get("position")[:]

                # The velocity and acceleration computation is based on an
                # inversion of Semi-Implicit Euler
                if loop == "mean":
                    vels.append((r[1:] - r[:-1]).mean((0, 1)))
                    accs.append((r[2:] + r[:-2] - 2 * r[1:-1]).mean((0, 1)))
                elif loop == "std":
                    centered_vel = r[1:] - r[:-1] - vel_mean
                    vels_sq.append(np.square(centered_vel).mean((0, 1)))
                    centered_acc = r[2:] + r[:-2] - 2 * r[1:-1] - acc_mean
                    accs_sq.append(np.square(centered_acc).mean((0, 1)))

            hf.close()
            print(f"Finished {loop} loop of {dataset_root} {split}!")

        if loop == "mean":
            vel_mean = np.stack(vels).mean(0)
            acc_mean = np.stack(accs).mean(0)
            print(f"vel_mean={vel_mean}, acc_mean={acc_mean}")
        elif loop == "std":
            vel_std = np.stack(vels_sq).mean(0) ** 0.5
            acc_std = np.stack(accs_sq).mean(0) ** 0.5
            print(f"vel_std={vel_std}, acc_std={acc_std}")

    # metadata
    with open(os.path.join(dataset_root, "metadata.json"), "r") as f:
        metadata_dict = json.load(f)

    metadata_dict["vel_mean"] = vel_mean.tolist()
    metadata_dict["vel_std"] = vel_std.tolist()

    metadata_dict["acc_mean"] = acc_mean.tolist()
    metadata_dict["acc_std"] = acc_std.tolist()

    with open(os.path.join(dataset_root, "metadata.json"), "w") as f:
        json.dump(metadata_dict, f)


def read_h5_demo(file_path):
    """Simple demonstation how to read the previously converted h5 file"""

    hf = h5py.File(f"{file_path[:-9]}.h5", "r")

    for key in hf.keys():
        r = hf[f"{key}/r"][:]
        print(key, r.shape, r.mean())

    hf.close()


def create_h5_with_first_taj_of_other_h5(
    src_path="GNS/data/WaterDropSample/train.h5",
    dst_path="GNS/data/WaterDropSample/train_first_traj.h5",
):
    """create a split_one_traj.h5 file from split.h5 with only one trajectoy"""

    src = h5py.File(src_path, "r")
    dst = h5py.File(dst_path, "w")

    particle_type = src["0000/particle_type"][:]
    position = src["0000/position"][:]

    dst.create_dataset("0000/particle_type", data=particle_type)
    dst.create_dataset(
        "0000/position", data=position, dtype=np.float32, compression="gzip"
    )

    src.close()
    dst.close()


def create_h5_with_first_n_frames_of_other_h5(
    src_root="/home/atoshev/data/TGV4k12",
    dst_root="/home/atoshev/data/TGV1k12",
    n_traj_steps=1000,
):
    """Create a smaller dataset by taking the first n_traj_steps of each trajectory"""

    os.makedirs(dst_root, exist_ok=True)

    for split in ["train", "valid", "test"]:
        src = h5py.File(f"{src_root}/{split}.h5", "r")
        dst = h5py.File(f"{dst_root}/{split}.h5", "w")

        for key, v in src.items():
            position = v.get("position")[: n_traj_steps + 1]
            particle_type = v.get("particle_type")[:]

            dst.create_dataset(
                f"{key}/position", data=position, dtype=np.float32, compression="gzip"
            )
            dst.create_dataset(f"{key}/particle_type", data=particle_type)

        dst.close()
        src.close()

    # metadata file
    with open(os.path.join(src_root, "metadata.json"), "r") as f:
        metadata_dict = json.load(f)

    metadata_dict["sequence_length"] = n_traj_steps

    with open(os.path.join(dst_root, "metadata.json"), "w") as f:
        json.dump(metadata_dict, f)

    compute_statistics_h5(dataset_root=dst_root)


if __name__ == "__main__":
    compute_statistics_h5("/home/atoshev/data/TGV1k12")
    # create_h5_with_first_taj_of_other_h5()
    # create_h5_with_first_n_frames_of_other_h5()
