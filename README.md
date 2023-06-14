# LagrangeBench: A Lagrangian Fluid Mechanics Benchmarking Suite

## Installation
Install the dependencies with Poetry (>1.5.0)
```
poetry install
```
Alternatively, a `requirements.txt` file is provided
```
pip install -r requirements.txt
```

## Dataset
The datasets are temporarily hosted on Google Drive. To download them use the `download_data.sh` shell script, either with the dataset name or "all". Namely
- __Taylor Green Vortex 2D__: `bash download_data.sh tgv_2d`
- __Reverse Poiseuille Flow 2D__: `bash download_data.sh rpf_2d`
- __Lid Driven Cavity 2D__: `bash download_data.sh ldc_2d`
- __Dam break 2D__: `bash download_data.sh dam_2d`
- __Taylor Green Vortex 3D__: `bash download_data.sh tgv_3d`
- __Reverse Poiseuille Flow 3D__: `bash download_data.sh rpf_3d`
- __Lid Driven Cavity 3D__: `bash download_data.sh ldc_3d`
- __All__: `bash download_data.sh all`

## Usage
Runs are based around YAML config files and cli arguments. By default, passed cli arguments will overwrite the YAML config.
When loading a model with `--model_dir` the correct config is automatically loaded and training is restarted.

For example, to start a _GNS_ run from scratch on the RPF 2D dataset use
```
python main.py --config configs/rpf_2d/gns.yaml
```

Some model presets are found in [configs/](/configs/).

## Adding new models
To add a new model create a subclass of [`models.BaseModel`](/lagrangebench/models/base.py) in the [`models`](/lagrangebench/models/) package that inherits, and then add it to the [`model_dict`](/lagrangebench/models/__init__.py#L12) dictionary.
