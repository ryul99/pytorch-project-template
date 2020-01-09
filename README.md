# Deep Learning Project Template for PyTorch
## Feature
- TensorBoardX
- checkpoint saving, resuming
- config with .yaml file

# Setup
## Install requirements
- python3
- `pip install -r requirements.txt`

## Config
- config is written in yaml file(default: `config/default.yaml`)

# Train
- `python trainer.py -c config/path/to/file -n model_name`

## Resume training from checkpoint
- `python trainer.py -c path/to/config/file -n model_name - p path/to/checkpoint/file`

# Reference
- https://github.com/mindslab-ai/voicefilter
