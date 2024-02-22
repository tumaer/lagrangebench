"""Default lagrangebench configs."""

import os.path as osp

from omegaconf import OmegaConf

# look for a "defaults.yaml" file in the same directory as this file
defaults = OmegaConf.load(osp.realpath(__file__).replace(".py", ".yaml"))
