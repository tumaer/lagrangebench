# LagrangeBench: A Lagrangian Fluid Mechanics Benchmarking Suite

## Installation
Install the dependencies with Poetry (>1.5.0)
```
poetry install
```
Alternatively, the `requirements.txt` file is provided
```
pip install -r requirements.txt
```

<!-- ## Dataset
### Taylor Green Vortex

```
sh download_data.sh tgv
```
### Reverse Poiseuille Flow

```
sh download_data.sh rpf
```
### Hookes Law (demo dataset)

```
sh download_data.sh hook
``` -->

## Usage
Runs are based around YAML config files and cli arguments. By default, passed cli arguments will overwrite the YAML config.
When loading a model with `--model_dir` the correct config is automatically loaded and training is restarted.

For example, to start a _SEGNN_ run from scratch on the TGV 3D dataset use
```
python main.py --config configs/tgv_3d/segnn.yaml
```

The files found in [configs/](/configs/) are the default baseline configurations.


## Adding new models
To add a new model create a subclass of [`models.BaseModel`](/lagrangebench/models/base.py) in the [`models`](/lagrangebench/models/) package that inherits, and then add it to the [`model_dict`](/lagrangebench/models/__init__.py#L12) dictionary.

<!-- ## Citing
This codebase was created by Artur Toshev and Gianluca Galletti. If you use our work in your research, please cite it:
```bibtex
@article{Toshev2023LagrangeBench,
    title={LagrangeBench: A Lagrangian Fluid Mechanics Benchmarking Suite},
    author={Artur P. Toshev and Gianluca Galletti, Fabian Fritz, Stefan Adami, Nikolaus A. Adams},
}
``` -->
