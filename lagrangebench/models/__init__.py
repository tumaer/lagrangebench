from typing import Callable, Tuple, Type

from .egnn import EGNN
from .gns import GNS
from .linear import Linear
from .painn import PaiNN
from .segnn import SEGNN

__all__ = ["get_model"]

model_dict = {
    "gns": GNS,
    "egnn": EGNN,
    "segnn": SEGNN,
    "linear": Linear,
    "painn": PaiNN,
}


def get_model(model_name: str, *args, **kwargs) -> Tuple[Callable, Type]:
    """Setup model based on args."""
    assert model_name in model_dict, f"Unknown model: {model_name}"

    model_class = model_dict[model_name]

    def model(x):
        return model_class.setup_model(*args, **kwargs)(x)

    return model, model_class
