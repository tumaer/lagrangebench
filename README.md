# Learning Lagrangian Fluid Mechanics with E(3)-Equivariant Graph Neural Networks

Official __Jax__ implementation with code and experiments of:

__Learning Lagrangian Fluid Mechanics with E(3)-Equivariant GNNs__<br>
Artur P. Toshev, Gianluca Galletti, Johannes Brandstetter, Stefan Adami and Nikolaus A. Adams.<br>
https://arxiv.org/abs/

<img src="assets/gsi.png" width="400">

__Abstract:__ We contribute to the vastly growing field of machine learning for engineering systems by demonstrating that equivariant graph neural networks have the potential to learn more accurate dynamic-interaction models than their non-equivariant counterparts. We benchmark two well-studied fluid-flow systems, namely 3D decaying Taylor-Green vortex and 3D reverse Poiseuille flow, and evaluate the models based on different performance measures, such as kinetic energy or Sinkhorn distance. In addition, we investigate different embedding methods of physical-information histories for equivariant models. We find that while currently being rather slow to train and evaluate, equivariant models with our proposed history embeddings learn more accurate physical interactions.


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

For example, to start an _HAE SEGNN (linear)_ run on the TGV dataset from scratch use
```
python main.py --config configs/hae_segnn_lin/tgv.yaml
```

The files found in [configs/](/configs/) are the configurations used in the paper experiments.


## Adding new models
To add a new model create a subclass of [`models.BaseModel`](/equisph/models/base.py) in the [`models`](/equisph/models/) package that inherits, and then add it to the [`model_dict`](/equisph/models/__init__.py#L12) dictionary.

## Citing
This codebase was created by Artur Toshev and Gianluca Galletti. If you use our work in your research, please cite it:
```bibtex
@article{Toshev2023Learning,
    title={Learning Lagrangian Fluid Mechanics with E(3)-Equivariant Graph Neural Networks},
    author={Artur P. Toshev and Gianluca Galletti and Johannes Brandstetter, Stefan Adami and Nikolaus A. Adams},
}
```
