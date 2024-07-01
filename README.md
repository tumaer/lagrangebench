<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://svgshare.com/i/11hG.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://svgshare.com/i/11hG.svg">
<img alt="LagrangeBench Logo: Lagrangian Fluid Mechanics Benchmarking Suite" src="https://svgshare.com/i/11hG.svg" width=550pt>
</picture>
<!-- [![Static Badge](https://img.shields.io/badge/docs-red?style=for-the-badge&logo=readthedocs)](https://lagrangebench.readthedocs.io/en/latest/index.html)
[![Static Badge](https://img.shields.io/badge/arxiv-blue?style=for-the-badge&logo=arxiv)](https://arxiv.org/abs/2309.16342)
[![Ruff Linting](https://github.com/tumaer/lagrangebench/actions/workflows/ruff.yml/badge.svg)](https://github.com/tumaer/lagrangebench/actions/workflows/ruff.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Downloads](https://static.pepy.tech/badge/lagrangebench)](https://pypi.org/project/lagrangebench/)
[![Python Versions](https://img.shields.io/pypi/pyversions/lagrangebench)](https://pypi.org/project/lagrangebench/) -->

[![Paper](http://img.shields.io/badge/paper-arxiv.2309.16342-B31B1B.svg)](https://arxiv.org/abs/2309.16342)
[![Docs](https://img.shields.io/readthedocs/lagrangebench/latest)](https://lagrangebench.readthedocs.io/en/latest/index.html)
[![PyPI - Version](https://img.shields.io/pypi/v/lagrangebench)](https://pypi.org/project/lagrangebench/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tumaer/lagrangebench/blob/main/notebooks/tutorial.ipynb)
[![Discord](https://img.shields.io/badge/Discord-%235865F2?logo=discord&logoColor=white)](https://discord.gg/Ds8jRZ78hU)

[![Tests](https://github.com/tumaer/lagrangebench/actions/workflows/tests.yml/badge.svg)](https://github.com/tumaer/lagrangebench/actions/workflows/tests.yml)
[![CodeCov](https://codecov.io/gh/tumaer/lagrangebench/graph/badge.svg?token=ULMGSY71R1)](https://codecov.io/gh/tumaer/lagrangebench)
[![License](https://img.shields.io/pypi/l/lagrangebench)](https://github.com/tumaer/lagrangebench/blob/main/LICENSE)

</div>

NeurIPS page with video and slides [here](https://neurips.cc/virtual/2023/poster/73681).

## Table of Contents

1. [**Installation**](#installation)
1. [**Usage**](#usage)
1. [**Datasets**](#datasets)
1. [**Pretrained Models**](#pretrained-models)
1. [**Directory Structure**](#directory-structure)
1. [**Contributing**](#contributing)
1. [**Citation**](#citation)

## Installation
### Standalone library
Install the core `lagrangebench` library from PyPi as
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install lagrangebench --extra-index-url=https://download.pytorch.org/whl/cpu
```

Note that by default `lagrangebench` is installed without JAX GPU support. For that follow the instructions in the [GPU support](#gpu-support) section.

### Clone
Clone this GitHub repository
```bash
git clone https://github.com/tumaer/lagrangebench.git
cd lagrangebench
```

Install the dependencies with __Poetry (>=1.6.0)__
```
poetry install --only main
```

Alternatively, a requirements file is provided. It directly installs the CUDA version of JAX.
```
pip install -r requirements_cuda.txt
```
For a CPU version of the requirements file, one could use `docs/requirements.txt`.

### GPU support
To run JAX on GPU, follow [Installing JAX](https://jax.readthedocs.io/en/latest/installation.html), or in general run
```bash
pip install -U "jax[cuda12]==0.4.29"
```

> Note: as of 27.06.2024, to make our GNN models **deterministic** on GPUs, you need to set `os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"`. However, all current models rely of `scatter_sum`, and this operation seems to be slower than running a normal for-loop in Python, when executed in deterministic mode, see [#17844](https://github.com/google/jax/issues/17844) and [#10674](https://github.com/google/jax/discussions/10674).


### MacOS
Currently, only the CPU installation works. You will need to change a few small things to get it going:
- Clone installation: in `pyproject.toml` change the torch version from `2.1.0+cpu` to `2.1.0`. Then, remove the `poetry.lock` file and run `poetry install --only main`.
- Configs: You will need to set `dtype=float32` and `train.num_workers=0`.

Although the current [`jax-metal==0.0.5` library](https://pypi.org/project/jax-metal/) supports jax in general, there seems to be a missing feature used by `jax-md` related to padding -> see [this issue](https://github.com/google/jax/issues/16366#issuecomment-1591085071).

## Usage
### Standalone benchmark library
A general tutorial is provided in the example notebook "Training GNS on the 2D Taylor Green Vortex" under `./notebooks/tutorial.ipynb` on the [LagrangeBench repository](https://github.com/tumaer/lagrangebench). The notebook covers the basics of LagrangeBench, such as loading a dataset, setting up a case, training a model from scratch and evaluating its performance.

### Running in a local clone (`main.py`)
Alternatively, experiments can also be set up with `main.py`, based on extensive YAML config files and cli arguments (check [`configs/`](configs/)). By default, the arguments have priority as 1) passed cli arguments, 2) YAML config and 3) [`defaults.py`](lagrangebench/defaults.py) (`lagrangebench` defaults).

When loading a saved model with `load_ckp` the config from the checkpoint is automatically loaded and training is restarted. For more details check the [`runner.py`](lagrangebench/runner.py) file.

**Train**

For example, to start a _GNS_ run from scratch on the RPF 2D dataset use
```
python main.py config=configs/rpf_2d/gns.yaml
```
Some model presets can be found in `./configs/`.

If `mode=all` is provided, then training (`mode=train`) and subsequent inference (`mode=infer`) on the test split will be run in one go.


**Restart training**

To restart training from the last checkpoint in `load_ckp` use
```
python main.py load_ckp=ckp/gns_rpf2d_yyyymmdd-hhmmss
```

**Inference**

To evaluate a trained model from `load_ckp` on the test split (`test=True`) use
```
python main.py load_ckp=ckp/gns_rpf2d_yyyymmdd-hhmmss/best rollout_dir=rollout/gns_rpf2d_yyyymmdd-hhmmss/best mode=infer test=True
```

If the default `eval.infer.out_type=pkl` is active, then the generated trajectories and a `metricsYYYY_MM_DD_HH_MM_SS.pkl` file will be written to `eval.rollout_dir`. The metrics file contains all `eval.infer.metrics` properties for each generated rollout.

### Notebooks
We provide three notebooks that show LagrangeBench functionalities, namely:
- [`tutorial.ipynb`](notebooks/tutorial.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tumaer/lagrangebench/blob/main/notebooks/tutorial.ipynb), with a general overview of LagrangeBench library, with training and evaluation of a simple GNS model,
- [`datasets.ipynb`](notebooks/datasets.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tumaer/lagrangebench/blob/main/notebooks/datasets.ipynb), with more details and visualizations of the datasets, and
- [`gns_data.ipynb`](notebooks/gns_data.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tumaer/lagrangebench/blob/main/notebooks/gns_data.ipynb), showing how to train models within LagrangeBench on the datasets from the paper [Learning to Simulate Complex Physics with Graph Networks](https://arxiv.org/abs/2002.09405).

## Datasets
The datasets are hosted on Zenodo under the DOI: [10.5281/zenodo.10021925](https://zenodo.org/doi/10.5281/zenodo.10021925). If a dataset is not found in `dataset.src`, the data is automatically downloaded. Alternatively, to manually download the datasets use the `download_data.sh` shell script, either with a specific dataset name or "all". Namely
- __Taylor Green Vortex 2D__: `bash download_data.sh tgv_2d datasets/`
- __Reverse Poiseuille Flow 2D__: `bash download_data.sh rpf_2d datasets/`
- __Lid Driven Cavity 2D__: `bash download_data.sh ldc_2d datasets/`
- __Dam break 2D__: `bash download_data.sh dam_2d datasets/`
- __Taylor Green Vortex 3D__: `bash download_data.sh tgv_3d datasets/`
- __Reverse Poiseuille Flow 3D__: `bash download_data.sh rpf_3d datasets/`
- __Lid Driven Cavity 3D__: `bash download_data.sh ldc_3d datasets/`
- __All__: `bash download_data.sh all datasets/`

## Pretrained Models
We provide pretrained model weights of our default GNS and SEGNN models on each of the 7 LagrangeBench datasets. You can download and run the checkpoints given below. In the table, we also provide the 20-step error measures on the full test split.

| Dataset | Model | MSE<sub>20</sub> | Sinkhorn | MSE<sub>E<sub>kin</sub></sub> |
| ------- |-------------------------------------------------------------------------------------- | ------ | ------ | ------ |
| 2D TGV  | [GNS-10-128](https://drive.google.com/file/d/19TO4PaFGcryXOFFKs93IniuPZKEcaJ37/view)  | 5.9e-6 | 3.2e-7 | 4.9e-7 |
|         | [SEGNN-10-64](https://drive.google.com/file/d/1llGtakiDmLfarxk6MUAtqj6sLleMQ7RL/view) | 4.4e-6 | 2.1e-7 | 5.0e-7 |
| 2D RPF  | [GNS-10-128](https://drive.google.com/file/d/1uYusVlP1ykUNuw58vo7Wss-xyTMmopAn/view)  | 4.0e-6 | 2.5e-7 | 2.7e-5 |
|         | [SEGNN-10-64](https://drive.google.com/file/d/108dZVWs2qxAvKiboeEBW-nIcv-aslhYP/view) | 3.4e-6 | 2.5e-7 | 1.4e-5 |
| 2D LDC  | [GNS-10-128](https://drive.google.com/file/d/1JvdsW0H6XrgC2_cwV3pP66cAm9j1-AXc/view)  | 1.5e-5 | 1.1e-6 | 6.1e-7 |
|         | [SEGNN-10-64](https://drive.google.com/file/d/1D_wgs2pD9pTXoJK76yi-R0K2tY_T6lPn/view) | 2.1e-5 | 3.7e-6 | 1.6e-5 |
| 2D DAM  | [GNS-10-128](https://drive.google.com/file/d/16bJz3VfSMxOG1II8kCg5DlzGhjvdip2p/view)  | 3.1e-5 | 1.4e-5 | 1.1e-4 |
|         | [SEGNN-10-64](https://drive.google.com/file/d/1_6rHxK81vzrdIMPtJ7rIkeoUgsTeKmSn/view) | 4.1e-5 | 2.3e-5 | 5.2e-4 |
| 3D TGV  | [GNS-10-128](https://drive.google.com/file/d/1DEkXxrebS9eyLSMlc_ztHrqlh29NgLXC/view)  | 5.8e-3 | 4.7e-6 | 4.8e-2 |
|         | [SEGNN-10-64](https://drive.google.com/file/d/1ivJnHTgfbQ0IJujc5O0CUoQNiGU4zi_d/view) | 5.0e-3 | 4.9e-6 | 3.9e-2 |
| 3D RPF  | [GNS-10-128](https://drive.google.com/file/d/1yo-qgShLd1sgS1u5zkMXdJvhuPBwEQQE/view)  | 2.1e-5 | 3.3e-7 | 1.8e-6 |
|         | [SEGNN-10-64](https://drive.google.com/file/d/1Qczh3Z_z0grTuRuPDHyiYLzV1zg7Liz9/view) | 1.7e-5 | 2.7e-7 | 1.7e-6 |
| 3D LDC  | [GNS-10-128](https://drive.google.com/file/d/1b3IIkxk5VcWiT8Oyqg1wex8-ZfJv2g_v/view)  | 4.1e-5 | 3.2e-7 | 1.9e-8 |
|         | [SEGNN-10-64](https://drive.google.com/file/d/1ZIg7FXc1l3C4ekc9WvVvjHEl5KKxOA_U/view) | 4.1e-5 | 2.9e-7 | 2.5e-8 |

To reproduce the numbers in the table, e.g., on 2D TGV with GNS, follow these steps:
```bash
# download the checkpoint (1) through the browser or 
# (2) using the file ID from the URL, i.e., for 2D TGV + GNS
gdown 19TO4PaFGcryXOFFKs93IniuPZKEcaJ37
# unzip the downloaded file `gns_tgv2d.zip`
python -c "import shutil; shutil.unpack_archive('gns_tgv2d.zip', 'gns_tgv2d')"
# evaluate the model on the test split
python main.py gpu=$GPU_ID mode=infer eval.test=True load_ckp=gns_tgv2d/best
```

## Directory structure
```
📦lagrangebench
 ┣ 📂case_setup     # Case setup manager
 ┃ ┣ 📜case.py      # CaseSetupFn class
 ┃ ┗ 📜features.py  # Feature extraction
 ┣ 📂data           # Datasets and dataloading utils
 ┃ ┣ 📜data.py      # H5Dataset class and specific datasets
 ┃ ┗ 📜utils.py
 ┣ 📂evaluate       # Evaluation and rollout generation tools
 ┃ ┣ 📜metrics.py
 ┃ ┣ 📜rollout.py
 ┃ ┗ 📜utils.py
 ┣ 📂models         # Baseline models
 ┃ ┣ 📜base.py      # BaseModel class
 ┃ ┣ 📜egnn.py
 ┃ ┣ 📜gns.py
 ┃ ┣ 📜linear.py
 ┃ ┣ 📜painn.py
 ┃ ┣ 📜segnn.py
 ┃ ┗ 📜utils.py
 ┣ 📂train          # Trainer method and training tricks
 ┃ ┣ 📜strats.py    # Training tricks
 ┃ ┗ 📜trainer.py   # Trainer method
 ┣ 📜defaults.py    # Default values
 ┣ 📜runner.py      # Runner wrapping training and inference
 ┗ 📜utils.py
```


## Contributing
Welcome! We highly appreciate [Github issues](https://github.com/tumaer/lagrangebench/issues) and [PRs](https://github.com/tumaer/lagrangebench/pulls).

You can also chat with us on [**Discord**](https://discord.gg/Ds8jRZ78hU).

### Contributing Guideline
If you want to contribute to this repository, you will need the dev dependencies, i.e.
install the environment with `poetry install` without the ` --only main` flag.
Then, we also recommend you install the pre-commit hooks
if you don't want to manually run `pre-commit run` before each commit. To sum up:

```bash
git clone https://github.com/tumaer/lagrangebench.git
cd lagrangebench
poetry install
source $PATH_TO_LAGRANGEBENCH_VENV/bin/activate

# install pre-commit hooks defined in .pre-commit-config.yaml
# ruff is configured in pyproject.toml
pre-commit install

# if you want to bump the version in both pyproject.toml and __init__.py, do
poetry self add poetry-bumpversion
poetry version patch  # or minor/major
```

After you have run `git add <FILE>` and try to `git commit`, the pre-commit hook will
fix the linting and formatting of `<FILE>` before you are allowed to commit.

You should also run the tests locally before creating a PR. Do this simply by:

```bash
# pytest is configured in pyproject.toml
pytest
```

### Clone vs Library
LagrangeBench can be installed by cloning the repository or as a standalone library. This offers more flexibility, but it also comes with its disadvantages: the necessity to implement some things twice. If you change any of the following things, make sure to update its counterpart as well:
- General setup in `lagrangebench/runner.py` and `notebooks/tutorial.ipynb`
- Configs in `configs/` and `lagrangebench/defaults.py`
- Zenodo URLs in `download_data.sh` and `lagrangebench/data/data.py`
- Dependencies in `pyproject.toml`, `requirements_cuda.txt`, and `docs/requirements.txt`
- Library version in `pyproject.toml` and `lagrangebench/__init__.py`


## Citation
The paper (at NeurIPS 2023 Datasets and Benchmarks) can be cited as:
```bibtex
@article{toshev2024lagrangebench,
  title={Lagrangebench: A lagrangian fluid mechanics benchmarking suite},
  author={Toshev, Artur and Galletti, Gianluca and Fritz, Fabian and Adami, Stefan and Adams, Nikolaus},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```

The associated datasets can be cited as:
```bibtex
@dataset{toshev_2024_10491868,
  author       = {Toshev, Artur P. and Adams, Nikolaus A.},
  title        = {LagrangeBench Datasets},
  month        = jan,
  year         = 2024,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.10491868},
  url          = {https://doi.org/10.5281/zenodo.10491868}
}
```


### Publications
The following further publications are based on the LagrangeBench codebase:

1. [Learning Lagrangian Fluid Mechanics with E(3)-Equivariant Graph Neural Networks (GSI 2023)](https://arxiv.org/abs/2305.15603), A. P. Toshev, G. Galletti, J. Brandstetter, S. Adami, N. A. Adams
2. [Neural SPH: Improved Neural Modeling of Lagrangian Fluid Dynamics (ICML 2024)](https://arxiv.org/abs/2402.06275), A. P. Toshev, J. A. Erbesdobler, N. A. Adams, J. Brandstetter
