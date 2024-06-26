"""Print a frame for visual inspection of the data."""

import argparse

import h5py
import matplotlib.pyplot as plt


def plot_frame(src_dir, frame):
    with h5py.File(src_dir, "r") as f:
        tag = f["00000/particle_type"][:]
        r = f["00000/position"][frame]

    plt.scatter(r[:, 0], r[:, 1], c=tag)
    plt.savefig(f"frame_{frame}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print a frame for visual inspection of the data."
    )
    parser.add_argument("--src_dir", type=str, help="Source directory.")
    parser.add_argument("--frame", type=int, help="Which frame to plot.")
    args = parser.parse_args()

    plot_frame(args.src_dir, args.frame)

    # Example:
    # python plot_frame.py --src_dir=datasets/2D_TGV_2500_10kevery100/train.h5 --frame=0
