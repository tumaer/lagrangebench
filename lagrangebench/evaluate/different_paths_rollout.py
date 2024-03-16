"""Evaluation and inference functions for generating rollouts."""
"""
Here, for every initial condition, we use different seeds and then  
"""

import os
import pickle
import time
from functools import partial
from typing import Callable, Iterable, List, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jax_md.partition as partition
from jax import jit, lax, random, vmap
from torch.utils.data import DataLoader

from lagrangebench.data import H5Dataset
from lagrangebench.data.utils import get_dataset_stats, numpy_collate
from lagrangebench.defaults import defaults
from lagrangebench.evaluate.metrics import MetricsComputer, MetricsDict
from lagrangebench.evaluate.utils import write_vtk
import os
from lagrangebench.utils import (
    ACDMConfig,
    broadcast_from_batch,
    broadcast_to_batch,
    get_kinematic_mask,
    load_haiku,
    set_seed,
)

#START HERE
