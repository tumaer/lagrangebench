import argparse

import numpy as np
import pandas as pd
import seaborn as sns

from gns_jax.utils import load_haiku


def get_attention_weights(model_dir: str) -> pd.DataFrame:
    """Returns the attribute embedding weights at every layer of SEGNN."""
    params, _, _, _ = load_haiku(model_dir)
    weights = {}
    for layer, tp_weighs in params.items():
        if "node_attribute_embedding" in layer:
            name = layer.split("/")[-1].replace("node_attribute_embedding_", "")
            weights[name] = np.vstack(tp_weighs.values()).flatten()
    return pd.DataFrame(weights).fillna(0.0).T


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    args = parser.parse_args()
    w = get_attention_weights(args.model_dir)

    # visualize the attention weights
    p = sns.heatmap(w, annot=False).set_title("Attention Weights")
    p.figure.savefig("attention_weights.png")
