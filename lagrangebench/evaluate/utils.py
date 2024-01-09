"""Utility functions for evaluation."""

import os
import pickle

import numpy as np


def write_vtk(data_dict, path):
    """Store a .vtk file for ParaView."""

    try:
        import pyvista
    except ImportError:
        raise ImportError("Please install pyvista to write VTK files.")

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


def pkl2vtk(src_path, dst_path=None):
    """Convert a rollout pickle file to a set of vtk files.

    Args:
        src_path (str): Source path to .pkl file.
        dst_path (str, optoinal): Destination directory path. Defaults to None.
            If None, then the vtk files are saved in the same directory as the pkl file.

    Example:
        pkl2vtk("rollout/test/rollout_0.pkl", "rollout/test_vtk")
        will create files rollout_0_0.vtk, rollout_0_1.vtk, etc. in the directory
        "rollout/test_vtk"
    """

    # set up destination directory
    if dst_path is None:
        dst_path = os.path.dirname(src_path)
    os.makedirs(dst_path, exist_ok=True)

    # load rollout
    with open(src_path, "rb") as f:
        rollout = pickle.load(f)

    file_prefix = os.path.join(dst_path, os.path.basename(src_path).split(".")[0])
    for k in range(rollout["predicted_rollout"].shape[0]):
        # predictions
        state_vtk = {
            "r": rollout["predicted_rollout"][k],
            "tag": rollout["particle_type"],
        }
        write_vtk(state_vtk, f"{file_prefix}_{k}.vtk")
        # ground truth reference
        state_vtk = {
            "r": rollout["ground_truth_rollout"][k],
            "tag": rollout["particle_type"],
        }
        write_vtk(state_vtk, f"{file_prefix}_ref_{k}.vtk")
