from .case_setup.case import case_builder
from .data import DAM2D, LDC2D, LDC3D, RPF2D, RPF3D, TGV2D, TGV3D, H5Dataset
from .evaluate import infer, infer_pde_refiner
from .models import EGNN, GNS, SEGNN, PaiNN
from .train.trainer import Trainer
from .utils import PushforwardConfig

__all__ = [
    "Trainer",
    "infer",
    "infer_pde_refiner",
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
    "PushforwardConfig",
]

__version__ = "0.0.1"
