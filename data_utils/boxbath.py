import h5py
import numpy as np


def find_bounds():

    # try to understand how deepmind dealt with boundaries by investigating Water-3D

    file_path = "/home/atoshev/code/sph-dataset-jax/GNS/data/Water-3D/valid.tfrecord"
    hf = h5py.File(f"{file_path[:-9]}.h5", "r")

    mins = np.zeros((len(hf.keys()), 3))
    for i, key in enumerate(hf.keys()):

        x = hf[key]["position"][-1]

        # lower bound (y)
        xx = x[
            (x[:, 0] > 0.2)
            * (x[:, 0] < 0.8)
            * (x[:, 1] < 0.13)
            * (x[:, 2] > 0.2)
            * (x[:, 2] < 0.8)
        ]
        my = xx.mean(0)[1]
        # print(xx.shape, "y_min", my)

        # x min
        xx = x[
            (x[:, 0] < 0.13)
            * (x[:, 1] > 0.17)
            * (x[:, 1] < 0.25)
            * (x[:, 2] > 0.2)
            * (x[:, 2] < 0.8)
        ]
        mx = xx.mean(0)[0]
        # print(xx.shape, "x_min", mx)

        # z min
        xx = x[
            (x[:, 0] > 0.2)
            * (x[:, 0] < 0.8)
            * (x[:, 1] > 0.17)
            * (x[:, 1] < 0.25)
            * (x[:, 2] < 0.13)
        ]
        mz = xx.mean(0)[2]
        # print(xx.shape, "z_min", mz)

        mins[i] = np.array([mx, my, mz])
    print(mins.mean(0))

    #########################
    # now do the same computation as for Water-3D but for BoxBath

    hf = h5py.File("/home/atoshev/data/BoxBath/test.h5", "r")
    stats = np.zeros((len(hf.keys()), 5))
    for i, key in enumerate(hf.keys()):

        x = hf[key]["position"][-1]

        # lower bound (y)
        xx = x[(x[:, 1] < 0.02)]
        miny = xx.mean(0)[1]

        # x min
        xx = x[(x[:, 0] < 0.0) * (x[:, 1] < 0.02)]
        minx = xx.mean(0)[0]

        # x max
        xx = x[(x[:, 0] > 1.19) * (x[:, 1] < 0.02)]
        maxx = xx.mean(0)[0]

        # z min
        xx = x[(x[:, 2] < 0.0) * (x[:, 1] < 0.02)]
        minz = xx.mean(0)[2]

        # z max
        xx = x[(x[:, 2] > 0.38) * (x[:, 1] < 0.02)]
        maxz = xx.mean(0)[2]

        stats[i] = np.array([minx, maxx, miny, minz, maxz])
    print(stats.mean(0))

    st = np.array(
        [
            [-0.00478244, 1.19998254, 0.01000073, -0.0049077, 0.38975055],
            [-0.00478245, 1.19997449, 0.01000075, -0.0049077, 0.38975479],
            [-0.00478244, 1.19996421, 0.01000102, -0.00490766, 0.38975915],
        ]
    )

    st_ave = 0.05 * st[0] + 0.05 * st[1] + 0.9 * st[2]  # valid, test, train
    print(st_ave)
    # array([-0.00478244,  1.19996564,  0.01000099, -0.00490766,  0.3897585 ])


if __name__ == "__main__":
    find_bounds()
