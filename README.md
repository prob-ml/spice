SPatial Inference of Communication Effects (SPICE)
========================================
<!-- ![tests](https://github.com/prob-ml/spatial/workflows/tests/badge.svg)
[![codecov](https://codecov.io/gh/prob-ml/spatial/branch/main/graph/badge.svg?token=98AQPGC96W)](https://codecov.io/gh/prob-ml/spatial) -->

# Installation

SPICE uses poetry as its package manager.

```
poetry install
poetry shell
pre-commit install
```

# Tutorial

Here, we briefly demonstrate how to use SPICE on a subset of Male, Excitatory cells from the [MERFISH Hypothalamus Dataset](https://datadryad.org/dataset/doi:10.5061/dryad.8t8s248).

### Configuration

SPICE uses [hydra](https://hydra.cc/docs/intro/) for configuring datasets, optimizers, training, testing, and models. Hydra builds upon OmegaConf to offer
CLI overrides, multiruns, syncing multiple configs together, managing outputs, etc.

Configurations can be found in the /config folder with each config file is represented by the following (GLOBAL STRUCTURE):

defaults:
  - _self_
  - optimizer:
  - datasets:
  - model:
  - training:
  - predict:

mode:

n_neighbors:

radius:

gpus:

paths:
  root:
  data: ${paths.root}/data
  models: ${paths.root}/models
  output: ${paths.root}/output

hydra:
  job:
    env_set:
      CUDA_VISIBLE_DEVICES:
  output_subdir:
  run:
    dir: .


### Training

spatial -m mode=train

### Training with CLI Hyperparameter Overrides

To

### Evaluation

spatial -m mode=predict
