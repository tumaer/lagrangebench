# LagrangeBench: A Lagrangian Fluid Mechanics Benchmarking Suite


## Installation (Pip)
If you only want to install the core `lagrangebench` library, use the Pip wheel.
```bash
pip install lagrangebench
```

## Installation (Clone)
1. Clone the GitHub repository
    ```bash
    git clone https://github.com/tumaer/lagrangebench.git
    cd lagrangebench
    ```
2. Install dependencies
    1. Install the dependencies with Poetry (>1.5.0)
        ```
        poetry install
        source .venv/bin/activate
        ```
    2. Alternatively, a `requirements.txt` file is provided
        ```
        pip install -r requirements.txt
        ```

## Usage
### Standalone benchmark library
A tutorial is provided in the example notebook "Training GNS on the 2D Taylor Green Vortex" under `./notebooks/tutorial.ipynb` on the [LagrangeBench Github repository](https://github.com/tumaer/lagrangebench). The notebook covers the basics of LagrangeBench, such as loading a dataset, setting up a case, training a model from scratch and evaluating it's performance.

### Running in a local clone (`main.py`)
Alternatively, runs are based around YAML config files and cli arguments. By default, passed cli arguments will overwrite the YAML config.
When loading a model with `--model_dir` the correct config is automatically loaded and training is restarted.

For example, to start a _GNS_ run from scratch on the RPF 2D dataset use
```
python main.py --config configs/rpf_2d/gns.yaml
```

Some model presets can be found in `./configs/`.


## Datasets
The datasets are temporarily hosted on Google Drive. When creating a new dataset instance the data is automatically downloaded. In alternative, to manually download them use the `download_data.sh` shell script, either with a specific dataset name or "all". Namely
- __Taylor Green Vortex 2D__: `bash download_data.sh tgv_2d`
- __Reverse Poiseuille Flow 2D__: `bash download_data.sh rpf_2d`
- __Lid Driven Cavity 2D__: `bash download_data.sh ldc_2d`
- __Dam break 2D__: `bash download_data.sh dam_2d`
- __Taylor Green Vortex 3D__: `bash download_data.sh tgv_3d`
- __Reverse Poiseuille Flow 3D__: `bash download_data.sh rpf_3d`
- __Lid Driven Cavity 3D__: `bash download_data.sh ldc_3d`
- __All__: `bash download_data.sh all`


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
