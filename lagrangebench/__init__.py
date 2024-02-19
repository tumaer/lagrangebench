from .case_setup.case import case_builder
from .config import cfg
from .data import DAM2D, LDC2D, LDC3D, RPF2D, RPF3D, TGV2D, TGV3D, H5Dataset
from .evaluate import infer
from .models import EGNN, GNS, SEGNN, PaiNN
from .train.trainer import Trainer

__all__ = [
    "Trainer",
    "infer",
    "case_builder",
    "GNS",
    "EGNN",
    "SEGNN",
    "PaiNN",
    "H5Dataset",
    "TGV2D",
    "TGV3D",
    "RPF2D",
    "RPF3D",
    "LDC2D",
    "LDC3D",
    "DAM2D",
    "cfg",
]

import toml

pyproject = toml.load("pyproject.toml")
__version__ = pyproject["tool"]["poetry"]["version"]
