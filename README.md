# LagrangeBench: A Lagrangian Fluid Mechanics Benchmarking Suite

## Installation
### Standalone library
Install the core `lagrangebench` library from PyPi as
```bash
pip install --extra-index-url=https://download.pytorch.org/whl/cpu lagrangebench
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

Alternatively, a `requirements.txt` file is provided. It directly installs the CUDA version of JAX.
```
pip install -r requirements_cuda.txt
```

### GPU support
To run JAX on GPU follow the [Jax CUDA guide](https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier), or in general run
```bash
pip install --upgrade jax[cuda11_pip]==0.4.18 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# or, for cuda 12
pip install --upgrade jax[cuda12_pip]==0.4.18 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Usage
### Standalone benchmark library
A general tutorial is provided in the example notebook "Training GNS on the 2D Taylor Green Vortex" under `./notebooks/tutorial.ipynb` on the [LagrangeBench repository](https://github.com/tumaer/lagrangebench). The notebook covers the basics of LagrangeBench, such as loading a dataset, setting up a case, training a model from scratch and evaluating it's performance.

Two additional notebooks are provided: `./notebooks/datasets.ipynb` with more details on the datasets, and `./notebooks/gns_data.ipynb` showing how to train models within LagrangeBench on DeepMind datasets.

### Running in a local clone (`main.py`)
Alternatively, experiments can also be set up with `main.py`, based around extensive YAML config files and cli arguments (check `configs/` and `experiments/configs.py`). By default, passed cli arguments will overwrite the YAML config. When loading a saved model with `--model_dir` the config from the checkpoint is automatically loaded and training is restarted.

For example, to start a _GNS_ run from scratch on the RPF 2D dataset use
```
python main.py --config configs/rpf_2d/gns.yaml
```

Some model presets can be found in `./configs/`.


## Datasets
The datasets are hosted on Zenodo under the DOI: [10.5281/zenodo.10021925](https://zenodo.org/doi/10.5281/zenodo.10021925). When creating a new dataset instance, the data is automatically downloaded. Alternatively, to manually download them use the `download_data.sh` shell script, either with a specific dataset name or "all". Namely
- __Taylor Green Vortex 2D__: `bash download_data.sh tgv_2d datasets/`
- __Reverse Poiseuille Flow 2D__: `bash download_data.sh rpf_2d datasets/`
- __Lid Driven Cavity 2D__: `bash download_data.sh ldc_2d datasets/`
- __Dam break 2D__: `bash download_data.sh dam_2d datasets/`
- __Taylor Green Vortex 3D__: `bash download_data.sh tgv_3d datasets/`
- __Reverse Poiseuille Flow 3D__: `bash download_data.sh rpf_3d datasets/`
- __Lid Driven Cavity 3D__: `bash download_data.sh ldc_3d datasets/`
- __All__: `bash download_data.sh all datasets/`


## Directory structure
```
📦lagrangebench
 ┣ 📂case_setup     # Case setup manager
 ┃ ┣ 📜case.py      # CaseSetupFn class
 ┃ ┣ 📜features.py  # Feature extraction
 ┃ ┗ 📜partition.py # Alternative neighbor list implementations
 ┣ 📂data           # Datasets and dataloading utils
 ┃ ┣ 📜data.py      # H5Dataset class and specific datasets
 ┃ ┗ 📜utils.py
 ┣ 📂evaluate       # Evaluation and rollout generation tools
 ┃ ┣ 📜metrics.py
 ┃ ┗ 📜rollout.py
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
 ┗ 📜utils.py
```


## Citation
The paper (at NeurIPS 2023 Datasets and Benchmarks) can be cited as:
```bibtex
@inproceedings{toshev2023lagrangebench,
    title      = {LagrangeBench: A Lagrangian Fluid Mechanics Benchmarking Suite},
    author     = {Artur P. Toshev and Gianluca Galletti and Fabian Fritz and Stefan Adami and Nikolaus A. Adams},
    year       = {2023},
    url        = {https://arxiv.org/abs/2309.16342},
    booktitle  = {37th Conference on Neural Information Processing Systems (NeurIPS 2023) Track on Datasets and Benchmarks},
}
```

The associated datasets can be cited as:
```bibtex
@dataset{toshev_2023_10021926,
  author       = {Toshev, Artur P. and Adams, Nikolaus A.},
  title        = {LagrangeBench Datasets},
  month        = oct,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {0.0.1},
  url          = {https://zenodo.org/doi/10.5281/zenodo.10021925}
  doi          = {10.5281/zenodo.10021925},
}```
