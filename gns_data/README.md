# Demonstration on how to train the GNS model on one of its original 2D datasets

> Check out the full notebook under [`notebooks/gns_data.ipynb`](../notebooks/gns_data.ipynb).

## Download data

```bash
mkdir -p /tmp/datasets
bash ./gns_data/download_dataset.sh WaterDrop /tmp/datasets
```

## Transform data from .tfrecord to .h5

First, you need the `tensorflow` and `tensorflow-datasets` libraries. We recommend installing these in a separate virtual environment to avoid CUDA version conflicts.

```bash
python3 -m venv venv_tf
venv_tf/bin/pip install tensorflow tensorflow-datasets
```

Then, transform the data via

```bash
./venv_tf/bin/python gns_data/tfrecord_to_h5.py --dataset-path=/tmp/datasets/WaterDrop
```

and train the usual way.
