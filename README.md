# Deep Learning Project Template for PyTorch

## Inspired by

- https://github.com/open-mmlab/mmsr
- https://github.com/mindslab-ai/voicefilter

## Feature

- TensorBoardX / [wandb](https://www.wandb.com/) support
- background generator is used ([reason of using background generator](https://github.com/IgorSusmelj/pytorch-styleguide/issues/5))
  - In Windows, background generator could not be supported. So if error occurs, set false to `use_background_generator` in config
- training state and network checkpoint saving, loading
- config with yaml file / easy dot-style access to config
- code lint / CI

## Setup

### Install requirements

- python3
- `pip install -r requirements.txt`

### Config

- config is written in yaml file(default: `config/default.yaml`)

### Code lint

1. `pip install -r requirements-dev.txt` for install develop dependencies (this requires python 3.6 and above because of black)

1. `pre-commit install` for adding pre-commit to git hook

## Train

- `python trainer.py -c config/path/to/file -n model_name`
