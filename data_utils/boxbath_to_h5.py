import os

import h5py
import numpy as np


def original_demo(dataset_root="GNS/data/BoxBath"):
    def load_data(data_names, path):
        hf = h5py.File(path, "r")
        data = []
        for i in range(len(data_names)):
            data.append(np.array(hf.get(data_names[i])))
        hf.close()
        return data

    data_names = ["positions", "velocities", "clusters"]

    data = load_data(data_names, dataset_root)

    rigid_particles_positions = data[0][:64]
    fluid_particles_positions = data[0][64:]

    print("rigid particles:", rigid_particles_positions.shape)
    print("fluid particles:", fluid_particles_positions.shape)

    # stats
    metadata_path = "GNS/data/BoxBath/stat.h5"
    stat = h5py.File(metadata_path)
    for k, v in stat.items():
        print(k, v[:])


def boxbath_to_packed_h5(dataset_root="GNS/data/BoxBath"):
    def load_h5(path):
        hf = h5py.File(path, "r")
        data = {}
        for k, v in hf.items():
            data[k] = v[:]
        hf.close()
        return data

    for split in ["train", "valid", "test"]:

        hf = h5py.File(f"{dataset_root}/{split}.h5", "w")

        if split == "test":
            split_path = os.path.join(dataset_root, "valid")
        else:
            split_path = os.path.join(dataset_root, split)

        traj_names = sorted(os.listdir(split_path), key=lambda x: int(x))

        if split == "valid":
            traj_names = traj_names[: (len(traj_names) // 2)]
        elif split == "test":
            traj_names = traj_names[(len(traj_names) // 2) :]

        for i, traj in enumerate(traj_names):
            traj_path = os.path.join(split_path, traj)
            frame_names = sorted(os.listdir(traj_path), key=lambda x: int(x[:-3]))
            assert len(frame_names) == 151, "Shape mismatch"

            position = np.zeros((151, 1024, 3))  # (time steps, particles, dim)
            for j, frame in enumerate(frame_names):
                frame_path = os.path.join(traj_path, frame)
                data = load_h5(frame_path)

                assert data["positions"].shape == (1024, 3), "Shape mismatch"
                position[j] = data["positions"]

            # tags {0: water, 1: solid wall, 2: moving wall, 3: rigid body}
            particle_type = np.where(np.arange(1024) < 64, 3, 0)

            traj_str = str(i).zfill(4)
            hf.create_dataset(f"{traj_str}/particle_type", data=particle_type)
            hf.create_dataset(
                f"{traj_str}/position",
                data=position,
                dtype=np.float32,
                compression="gzip",
            )

        hf.close()
        print("Finished boxbath_to_packed_h5!")


def compute_statistics_h5(dataset_root="GNS/data/BoxBath"):
    """Compute the mean and std of a h5 dataset files"""

    vels, accs = [], []
    vels_sq, accs_sq = [], []
    for loop in ["mean", "std"]:
        for split in ["train", "valid", "test"]:
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

        if loop == "mean":
            vel_mean = np.stack(vels).mean(0)
            acc_mean = np.stack(accs).mean(0)
            print(f"vel_mean={vel_mean}, acc_mean={acc_mean}")
        elif loop == "std":
            vel_std = np.stack(vels_sq).mean(0) ** 0.5
            acc_std = np.stack(accs_sq).mean(0) ** 0.5
            print(f"vel_std={vel_std}, acc_std={acc_std}")


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


if __name__ == "__main__":
    # boxbath_to_packed_h5()
    # compute_statistics_h5()
    # create_h5_with_first_taj_of_other_h5()
    pass
