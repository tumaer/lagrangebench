.. LagrangeBench documentation master file, created by
   sphinx-quickstart on Fri Aug 18 19:44:58 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

LagrangeBench
=============

.. image:: https://drive.google.com/thumbnail?id=1rP0pf1KL8iGbly0tA0qthUE_tMDv_9Jp&sz=w1000
   :alt: rpf2d.gif

.. image:: https://drive.google.com/thumbnail?id=1BMGkHj9EYMGUOdsE5QwiJWCTvDNqveHc&sz=w1000
   :alt: rpf3d.gif


What is ``LagrangeBench``?
--------------------------

LagrangeBench is a machine learning benchmarking suite for **Lagrangian particle
problems** based on the `JAX <https://jax.readthedocs.io/>`_ library. It provides:

- **Data loading and preprocessing** utilities for particle data.
- Three different **neighbors search routines**: (a) original JAX-MD implementation, (b)
  memory efficient version of the JAX-MD implementation, and (c) a wrapper around the
  matscipy implementation allowing to handle variable number of particles.
- JAX reimplementation of established **graph neural networks**: GNS, SEGNN, EGNN, PaiNN.
- **Training strategies** including random-walk additive noise and the pushforward trick.
- Evaluation tools consisting of **rollout generation** and different **error metrics**:
  position MSE, kinetic energy MSE, and Sinkhorn distance for the particle distribution.


.. note::

   For more details on LagrangeBench usage check out our `tutorials <pages/tutorial.html>`_.



Data loading and preprocessing
------------------------------

First, we create a dataset class based on ``torch.utils.data.Dataset``.
We then initialize a ``CaseSetupFn`` object taking care of the neighbors search,
preprocessing, and time integration.

.. code-block:: python

   import lagrangebench

   # Load data
   data_train = lagrangebench.RPF2D("train")
   data_valid = lagrangebench.RPF2D("valid", extra_seq_length=20)
   data_test = lagrangebench.RPF2D("test", extra_seq_length=20)

   # Case setup (preprocessing and graph building)
   bounds = np.array(data_train.metadata["bounds"])
   box = bounds[:, 1] - bounds[:, 0]
   case = lagrangebench.case_builder(
      box=box,
      metadata=data_train.metadata,
      input_seq_length=6,
   )


Models
------

Initialize a GNS model.

.. code-block:: python

   import haiku as hk

   def gns(x):
      return lagrangebench.models.GNS(
         particle_dimension=data_train.metadata["dim"],
         latent_size=16,
         blocks_per_step=2,
         num_mp_steps=4,
         particle_type_embedding_size=8,
      )(x)

   gns = hk.without_apply_rng(hk.transform_with_state(gns))


Training
--------

The ``Trainer`` provides a convenient way to train a model.

.. code-block:: python

   trainer = lagrangebench.Trainer(
      model=gns,
      case=case,
      data_train=data_train,
      data_valid=data_valid,
      cfg_eval={"n_rollout_steps": 20, "train": {"metrics": ["mse"]}},
      input_seq_length=6
   )

   # Train for 25000 steps
   params, state, _ = trainer.train(step_max=25000)


Evaluation
----------

When training is done, we can evaluate the model on the test set.

.. code-block:: python

   metrics = lagrangebench.infer(
      gns,
      case,
      data_test,
      params,
      state,
      cfg_eval_infer={"metrics": ["mse", "sinkhorn", "e_kin"]},
      n_rollout_steps=20,
   )


.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   pages/tutorial
   pages/defaults
   pages/baselines

.. toctree::
   :maxdepth: 2
   :caption: API

   pages/data
   pages/case_setup
   pages/models
   pages/train
   pages/evaluate
   pages/utils
