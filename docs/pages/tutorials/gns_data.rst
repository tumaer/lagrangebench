Working with other datasets
===========================

We demonstrate how to train a GNN on one of the datasets provided by
DeepMind along with the paper [1]. We divide this notebook into three
parts:

-  Download and preprocess
-  Inspect the data
-  GNN training

[1] - Sanchez-Gonzalez et al., `“Learning to Simulate Complex Physics
with Graph Networks” <https://arxiv.org/abs/2002.09405>`__, ICLR 2020

.. code:: ipython3

    import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    
    import lagrangebench
    import haiku as hk
    import numpy as np
    import matplotlib.pyplot as plt



Download and preprocess
-----------------------

The easiest was to download and preprocess the datasets from the GNS
paper is by following the instructions in
`gns_data.README.md <https://github.com/tumaer/lagrangebench/blob/aaf7c274d50ceebd001ce3a93a3160e40a8b04bc/gns_data/README.md>`.
For our demonstration, we choose the 2D WaterDrop dataset.

Note: the size of this dataset is around 4.5GB.

.. code:: ipython3

    !mkdir -p ./datasets
    !bash ../gns_data/download_dataset.sh WaterDrop ./datasets

To avoid conflicting library dependencies, we recommend installing a
second virtual environment only for the preprocessing. It will have the
lightweight CPU version of TensorFlow, which is needed to open the
original datasets.

.. code:: ipython3

    !python3 -m venv venv_tf
    !venv_tf/bin/pip install tensorflow tensorflow-datasets

Finally, we transform the ``*.tfrecord`` files to our ``*.h5`` format.
We automatically add two further features to the ``metadata.json`` file,
namely:

-  ``num_particles_max``: needed for jitability (via padding)
-  ``periodic_boundary_conditions``: specifying the type of boundary
   conditions per dimension

Note: This might take a few minutes and will double the space taken by
the dataset.

.. code:: ipython3

    !venv_tf/bin/python ../gns_data/tfrecord_to_h5.py --dataset-path=./datasets/WaterDrop

If everything worked fine and you see the ``*.h5`` files, you can remove
the ``*.tfrecord`` files and the virtual environment.

.. code:: ipython3

    !rm ./datasets/WaterDrop/*.tfrecord
    !rm -r venv_tf

Inspect the data
----------------

Because the number of particles in this dataset varies, we are forced to
use ``matscipy`` as the neighbors search implementation. Matscipy runs
on the CPU and can efficiently handle systems with up to milions of
particles. To integrate this function into our jit-able codebase, we use
``jax.pure_callback()`` and pad the non-existing particle entries up to
``num_particles_max``.

.. code:: ipython3

    data_train = lagrangebench.data.H5Dataset(
        split="train", 
        dataset_path="./datasets/WaterDrop",
        name="waterdrop2d",
        nl_backend="matscipy"
    )
    
    data_valid = lagrangebench.data.H5Dataset(
        split="valid", 
        dataset_path="./datasets/WaterDrop",
        name="waterdrop2d",
        split_valid_traj_into_n=38, # from [1], Appendix B.1, trajectory length is 1000
        is_rollout=True,
        nl_backend="matscipy"
    )
    
    print(
        f"This is a {data_train.metadata['dim']}D dataset from {data_train.dataset_path}.\n"
        f"Train frames have shape {data_train[0][0].shape} (n_nodes, seq_len, xy pos).\n"
        f"Val frames have shape {data_valid[0][0].shape} (n_nodes, rollout, xy pos).\n"
        f"And particle types have shape {data_train[0][1].shape} (n_nodes,).\n"
        f"Total of {len(data_train)} train frames and {len(data_valid)} val frames.\n"
    )


.. parsed-literal::

    This is a 2D dataset from ./datasets/WaterDrop.
    Train frames have shape (1108, 7, 2) (n_nodes, seq_len, xy pos).
    Val frames have shape (1108, 26, 2) (n_nodes, rollout, xy pos).
    And particle types have shape (1108,) (n_nodes,).
    Total of 994000 train frames and 1140 val frames.
    


Visualize slices from the first trajectory

.. code:: ipython3

    
    bounds = np.array(data_train.metadata["bounds"])
    
    # from [1], Appendix B.1 we know that the trajectory length is 1000
    # If we take indices < 994, we will see samples from the first trajectory
    for j in [0, 500, 993]:
        sample = data_train[j]
        
        # visualize 5 consecutive frames
        fig, axs = plt.subplots(1, 5, figsize=(20, 4))
        for i in range(5):
            mask = sample[1] != -1
            axs[i].scatter(sample[0][mask, i, 0], sample[0][mask, i, 1], s=1)
            axs[i].set_xlim(bounds[0])
            axs[i].set_ylim(bounds[1])
        plt.show()



.. image:: media/gns_data_13_0.png



.. image:: media/gns_data_13_1.png



.. image:: media/gns_data_13_2.png


Here we visualize random frames from the dataset.

.. code:: ipython3

    np.random.seed(42)
    frame_nums = np.random.randint(0, len(data_train), 15)
    
    # visualize 5 consecutive frames
    fig, axs = plt.subplots(3, 5, figsize=(20, 12))
    for ax, i in zip(axs.flatten(), frame_nums):
        sample = data_train[i]
        mask = sample[1] != -1
        
        ax.scatter(sample[0][mask, 0, 0], sample[0][mask, 0, 1], s=1, 
                   label=f"seed={i}, N={np.sum(mask)}")
        ax.legend()
        ax.set_xlim(bounds[0])
        ax.set_ylim(bounds[1])
    plt.show()



.. image:: media/gns_data_15_0.png


GNN training
------------

This works as usual. See
```notebooks/tutorial.ipynb`` <./tutorial.ipynb>`__ for more details.

.. code:: ipython3

    gns, _ = lagrangebench.get_model(
        "gns", data_train.metadata, latent_dim=16, num_mp_steps=4, num_mlp_layers=2
    )
    gns = hk.without_apply_rng(hk.transform_with_state(gns))
    
    noise_std = 1e-5
    bounds = np.array(data_train.metadata["bounds"])
    box = bounds[:, 1] - bounds[:, 0]
    
    case = lagrangebench.case_builder(
        box=box,  
        metadata=data_train.metadata,
        input_seq_length=6,
        isotropic_norm=False,
        noise_std=noise_std,
    )
    
    trainer = lagrangebench.Trainer(
        model=gns,
        case=case,
        dataset_train=data_train,
        dataset_eval=data_valid,
        noise_std=noise_std,
        metrics=["mse"],
        n_rollout_steps=20,
        eval_n_trajs=2,
        lr_start=5e-4,
        log_steps=10,
        eval_steps=50,
    )

.. code:: ipython3

    params, state, _ = trainer(step_max=101)


.. parsed-literal::

    Reallocate neighbors list (2, 170875) at step 0
    To list (2, 442352)
    000, train/loss: 5.78215.
    Reallocate neighbors list (2, 442352) at step 3
    To list (2, 684860)
    Reallocate neighbors list (2, 684860) at step 6
    To list (2, 915872)
    010, train/loss: 0.13905.
    020, train/loss: 0.38902.
    030, train/loss: 0.06267.
    040, train/loss: 0.05162.
    050, train/loss: 12.60960.
    {'val/loss': 7.373876087513054e-06, 'val/mse5': 5.798498570186439e-08, 'val/mse10': 6.175979763156647e-07, 'val/stdloss': 1.8027731130132452e-07, 'val/stdmse5': 2.589549907838773e-09, 'val/stdmse10': 3.496310796435864e-08}
    060, train/loss: 0.13755.
    070, train/loss: 0.72238.
    080, train/loss: 0.10497.
    090, train/loss: 0.18601.
    100, train/loss: 0.56957.
    {'val/loss': 6.617287453991594e-06, 'val/mse5': 5.17967393420804e-08, 'val/mse10': 5.519495687167364e-07, 'val/stdloss': 2.460046744090505e-07, 'val/stdmse5': 1.4909034007359878e-09, 'val/stdmse10': 2.3165881657405407e-08}

