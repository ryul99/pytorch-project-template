# Deep Learning Project Template for PyTorch

## Inspired by

- https://github.com/open-mmlab/mmsr
- https://github.com/mindslab-ai/voicefilter

## Feature

- TensorBoardX
- checkpoint saving, resuming
- config with yaml file
- dot-access to config
- code lint / CI

## Setup

### Install requirements

- python3
- `pip install -r requirements.txt`

### Config

- config is written in yaml file(default: `config/default.yaml`)

### Code lint

1. `pip install -r requirements-dev.txt` for install develop dependencies

1. `pre-commit install` for adding pre-commit to git hook

## Train

- `python trainer.py -c config/path/to/file -n model_name`

### Resume training from checkpoint

- `python trainer.py -c path/to/config/file -n model_name - p path/to/checkpoint/file`
