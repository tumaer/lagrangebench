import argparse
import os
from typing import Dict

import yaml


def cli_arguments() -> Dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="Path to the config yaml.")
    parser.add_argument(
        "--model", type=str, choices=["gns", "segnn", "lin"], help="Model name."
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size.")
    parser.add_argument(
        "--lr_start", type=float, default=1e-4, help="Starting learning rate."
    )
    parser.add_argument(
        "--lr_final", type=float, default=1e-6, help="Learning rate after decay."
    )
    parser.add_argument(
        "--lr_decay_rate", type=float, default=0.1, help="Learning rate decay."
    )
    parser.add_argument(
        "--lr_decay_steps", type=int, default=5e6, help="Learning rate decay steps."
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=6.7e-4,
        help="Additive noise standard deviation.",
    )
    parser.add_argument(
        "--input_seq_length",
        type=int,
        default=6,
        help="Input position sequence length.",
    )
    parser.add_argument(
        "--num_mp_steps", type=int, default=10, help="Number of message passing layers."
    )
    parser.add_argument(
        "--num_mlp_layers", type=int, default=2, help="Number of MLP layers."
    )
    parser.add_argument(
        "--latent_dim", type=int, default=128, help="Hidden layer dimension."
    )

    parser.add_argument(
        "--magnitude",
        action="store_true",
        help="Whether to include velocity magnitudes in node features.",
    )

    parser.add_argument(
        "--log_norm",
        default="none",
        choices=["none", "input", "output", "both"],
        help="Logarithmic normalization of input and/or output",
    )

    # segnn arguments
    parser.add_argument(
        "--lmax-attributes", type=int, default=1, help="Maximum degree of attributes."
    )
    parser.add_argument(
        "--lmax-hidden", type=int, default=1, help="Maximum degree of hidden layers."
    )
    parser.add_argument(
        "--norm",
        type=str,
        default="instance",
        choices=["instance", "batch", "none"],
        help="Normalisation type.",
    )
    parser.add_argument(
        "--velocity_aggregate",
        type=str,
        default="avg",
        choices=["avg", "sum", "last"],
        help="Velocity aggregation function for node attributes.",
    )
    parser.add_argument(
        "--attribute_mode",
        type=str,
        default="add",
        choices=["add", "concat", "velocity"],
        help="How to combine node attributes.",
    )

    return vars(parser.parse_args())


class NestedLoader(yaml.SafeLoader):
    """Load yaml files with nested includes."""

    def __init__(self, stream):
        self._root = os.path.split(stream.name)[0]
        super().__init__(stream)

    def get_single_data(self):
        parent = {}
        config = super().get_single_data()
        if "includes" in config and (included := config["includes"]):
            del config["includes"]
            with open(os.path.join(self._root, included), "r") as f:
                parent = yaml.load(f, NestedLoader)
        return {**parent, **config}
