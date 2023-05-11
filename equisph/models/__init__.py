from argparse import Namespace
from typing import Callable, Tuple, Type

from .gns import GNS
from .haesegnn import HAESEGNN
from .linear import Linear
from .segnn import SEGNN

__all__ = ["get_model"]

model_dict = {
    "gns": GNS,
    "segnn": SEGNN,
    "hae_segnn": HAESEGNN,
    "linear": Linear,
}


def get_model(args: Namespace) -> Tuple[Callable, Type]:
    """Setup model based on args."""
    assert args.config.model in model_dict, f"Unknown model: {args.config.model}"

    model_class = model_dict[args.config.model]

    def model(x):
        return model_class.setup_model(args)(x)

    return model, model_class
