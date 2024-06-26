"""Script for generating ML datasets from h5 simulation frames"""

import argparse
import json
import os

import h5py
import numpy as np
from jax import vmap
from jax_sph.io_state import read_h5, write_h5
from jax_sph.jax_md import space
from omegaconf import OmegaConf


def write_h5_frame_for_visualization(state_dict, file_path_h5):
    path_file_vis = os.path.join(file_path_h5[:-3] + "_vis.h5")
    print("writing to", path_file_vis)
    write_h5(state_dict, path_file_vis)
    print("done")


def single_h5_files_to_h5_dataset(args):
    """Transform a set of .h5 files to a single .h5 dataset file

    Args:
        src_dir: source directory containing other directories, each with .h5 files
            corresponding to a trajectory
        dst_dir: destination directory where three files will be written: train.h5,
            valid.h5, and test.h5
        split: string of three integers separated by underscores, e.g. "80_10_10"
    """

    os.makedirs(args.dst_dir, exist_ok=True)

    # list only directories in a root with files and directories
    dirs = os.listdir(args.src_dir)
    dirs = [d for d in dirs if os.path.isdir(os.path.join(args.src_dir, d))]
    # order by seed value
    dirs = sorted(dirs, key=lambda x: int(x.split("_")[3]))

    splits_array = np.array([int(s) for s in args.split.split("_")])
    splits_sum = splits_array.sum()

    if len(dirs) == 1:  # split one long trajectory into train, valid, and test
        files = os.listdir(os.path.join(args.src_dir, dirs[0]))
        files = [f for f in files if (".h5" in f)]
        files = sorted(files, key=lambda x: int(x.split("_")[1][:-3]))
        files = files[args.skip_first_n_frames :: args.slice_every_nth_frame]

        num_eval = np.ceil(splits_array[1] / splits_sum * len(files)).astype(int)
        # at least one validation and one testing trajectory
        splits_trajs = np.cumsum([0, len(files) - 2 * num_eval, num_eval, num_eval])

        num_trajs_train = num_trajs_test = 1

        sequence_length_train, sequence_length_test = splits_trajs[1] - 1, num_eval - 1
    else:  # multiple trajectories
        num_eval = np.ceil(splits_array[1] / splits_sum * len(dirs)).astype(int)
        # at least one validation and one testing trajectory
        splits_trajs = np.cumsum([0, len(dirs) - 2 * num_eval, num_eval, num_eval])

        num_trajs_train, num_trajs_test = len(dirs) - 2 * num_eval, num_eval

        # seqience_length should be after subsampling every nth trajectory
        # and "-1" because of the last target position (see GNS dataset format)
        files_per_traj = len(os.listdir(os.path.join(args.src_dir, dirs[0])))
        sequence_length_train = sequence_length_test = files_per_traj - 1

    for i, split in enumerate(["train", "valid", "test"]):
        hf = h5py.File(os.path.join(args.dst_dir, f"{split}.h5"), "w")

        if len(dirs) == 1:  # one long trajectory
            position = []
            traj_path = os.path.join(args.src_dir, dirs[0])

            for j, filename in enumerate(files[splits_trajs[i] : splits_trajs[i + 1]]):
                file_path_h5 = os.path.join(traj_path, filename)
                state = read_h5(file_path_h5, array_type="numpy")
                r = state["r"]
                tag = state["tag"]

                if "ldc" in args.src_dir.lower():  # remove outer walls in lid-driven
                    L, H = 1.0, 1.0
                    cfg = OmegaConf.load(os.path.join(traj_path, "config.yaml"))
                    mask_bottom = np.where(r[:, 1] < 2 * cfg.case.dx, False, True)
                    mask_lid = np.where(r[:, 1] > H + 4 * cfg.case.dx, False, True)
                    mask_left = np.where(
                        ((r[:, 0] < 2 * cfg.case.dx) * (tag == 1)), False, True
                    )
                    mask_right = np.where(
                        (r[:, 0] > L + 4 * cfg.case.dx) * (tag == 1), False, True
                    )
                    mask = mask_bottom * mask_lid * mask_left * mask_right

                    r = r[mask]
                    tag = tag[mask]

                if args.is_visualize:
                    write_h5_frame_for_visualization({"r": r, "tag": tag}, file_path_h5)
                position.append(r)

            position = np.stack(position)  # (time steps, particles, dim)
            particle_type = tag  # (particles,)

            traj_str = "00000"
            hf.create_dataset(f"{traj_str}/particle_type", data=particle_type)
            hf.create_dataset(
                f"{traj_str}/position",
                data=position,
                dtype=np.float32,
                compression="gzip",
            )

        else:  # multiple trajectories
            for j, dir in enumerate(dirs[splits_trajs[i] : splits_trajs[i + 1]]):
                traj_path = os.path.join(args.src_dir, dir)
                files = os.listdir(traj_path)
                files = [f for f in files if (".h5" in f)]
                files = sorted(files, key=lambda x: int(x.split("_")[1][:-3]))
                files = files[args.skip_first_n_frames :: args.slice_every_nth_frame]

                position = []
                for k, filename in enumerate(files):
                    file_path_h5 = os.path.join(traj_path, filename)
                    state = read_h5(file_path_h5, array_type="numpy")
                    r = state["r"]
                    tag = state["tag"]

                    if "db" in args.src_dir.lower():  # remove outer walls in dam break
                        L, H = 5.366, 2.0
                        cfg = OmegaConf.load(os.path.join(traj_path, "config.yaml"))
                        mask_bottom = np.where(r[:, 1] < 2 * cfg.sase.dx, False, True)
                        mask_lid = np.where(r[:, 1] > H + 4 * cfg.case.dx, False, True)
                        mask_left = np.where(
                            ((r[:, 0] < 2 * cfg.case.dx) * (tag == 1)), False, True
                        )
                        mask_right = np.where(
                            (r[:, 0] > L + 4 * cfg.case.dx) * (tag == 1), False, True
                        )
                        mask = mask_bottom * mask_lid * mask_left * mask_right

                        r = r[mask]
                        tag = tag[mask]

                    if args.is_visualize:
                        write_h5_frame_for_visualization(
                            {"r": r, "tag": tag}, file_path_h5
                        )
                    position.append(r)
                position = np.stack(position)  # (time steps, particles, dim)
                particle_type = tag  # (particles,)

                traj_str = str(j).zfill(5)
                hf.create_dataset(f"{traj_str}/particle_type", data=particle_type)
                hf.create_dataset(
                    f"{traj_str}/position",
                    data=position,
                    dtype=np.float32,
                    compression="gzip",
                )

        hf.close()
        print(f"Finished {args.src_dir} {split} with {j+1} entries!")
        print(f"Sample positions shape {position.shape}")

    # metadata
    # Compatible with the lagrangebench metadata.json files
    cfg = OmegaConf.load(os.path.join(traj_path, "config.yaml"))

    metadata = {
        "case": cfg.case.name.upper(),
        "solver": cfg.solver.name,
        "density_evolution": cfg.solver.density_evolution,
        "dim": cfg.case.dim,
        "dx": cfg.case.dx,
        "dt": cfg.solver.dt,
        "t_end": cfg.solver.t_end,
        "viscosity": cfg.case.viscosity,
        "p_bg_factor": cfg.eos.p_bg_factor,
        "g_ext_magnitude": cfg.case.g_ext_magnitude,
        "artificial_alpha": cfg.solver.artificial_alpha,
        "free_slip": cfg.solver.free_slip,
        "write_every": cfg.io.write_every,
        "is_bc_trick": cfg.solver.is_bc_trick,
        "sequence_length_train": int(sequence_length_train),
        "num_trajs_train": int(num_trajs_train),
        "sequence_length_test": int(sequence_length_test),
        "num_trajs_test": int(num_trajs_test),
        "num_particles_max": cfg.case.num_particles_max,
        "periodic_boundary_conditions": list(cfg.case.pbc),
        "bounds": np.array(cfg.case.bounds).tolist(),
    }
    x = 1.45 * cfg.case["dx"]  # around 1.5 dx
    x = np.format_float_positional(
        x, precision=2, unique=False, fractional=False, trim="k"
    )
    metadata["default_connectivity_radius"] = float(x)

    with open(os.path.join(args.dst_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)


def compute_statistics_h5(args):
    """Compute the mean and std of a h5 dataset files"""

    # metadata
    with open(os.path.join(args.dst_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)

    # apply PBC in all directions or not at all
    if np.array(metadata["periodic_boundary_conditions"]).any():
        box = np.array(metadata["bounds"])
        box = box[:, 1] - box[:, 0]
        displacement_fn, _ = space.periodic(side=box)
    else:
        displacement_fn, _ = space.free()

    displacement_fn_sets = vmap(vmap(displacement_fn, in_axes=(0, 0)))

    vels, accs = [], []
    vels_sq, accs_sq = [], []
    vel_mean = acc_mean = 0.0  # to fix "F821 Undefined name ..." ruff error
    for loop in ["mean", "std"]:
        for split in ["train", "valid", "test"]:
            hf = h5py.File(os.path.join(args.dst_dir, f"{split}.h5"), "r")

            for _, v in hf.items():
                tag = v.get("particle_type")[:]
                r = v.get("position")[:][:, tag == 0]  # only fluid ("0") particles

                # The velocity and acceleration computation is based on an
                # inversion of Semi-Implicit Euler
                vel = displacement_fn_sets(r[1:], r[:-1])
                if loop == "mean":
                    vels.append(vel.mean((0, 1)))
                    accs.append((vel[1:] - vel[:-1]).mean((0, 1)))
                elif loop == "std":
                    centered_vel = vel - vel_mean
                    vels_sq.append(np.square(centered_vel).mean((0, 1)))
                    centered_acc = vel[1:] - vel[:-1] - acc_mean
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

    # stds should not be 0. If they are, set them to 1.
    vel_std = np.where(vel_std < 1e-7, 1, vel_std)
    acc_std = np.where(acc_std < 1e-7, 1, acc_std)

    metadata["vel_mean"] = vel_mean.tolist()
    metadata["vel_std"] = vel_std.tolist()

    metadata["acc_mean"] = acc_mean.tolist()
    metadata["acc_std"] = acc_std.tolist()

    with open(os.path.join(args.dst_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str)
    parser.add_argument("--dst_dir", type=str)
    parser.add_argument("--split", type=str, help="E.g. 3_1_1")
    parser.add_argument("--skip_first_n_frames", type=int, default=0)
    parser.add_argument("--slice_every_nth_frame", type=int, default=1)
    parser.add_argument("--is_visualize", action="store_true")
    args = parser.parse_args()

    single_h5_files_to_h5_dataset(args)
    compute_statistics_h5(args)
