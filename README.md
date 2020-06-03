# Deep Learning Project Template for PyTorch

## Inspired by

- https://github.com/open-mmlab/mmsr
- https://github.com/mindslab-ai/voicefilter

## Feature

- TensorBoardX / [wandb](https://www.wandb.com/) support
- Background generator is used ([reason of using background generator](https://github.com/IgorSusmelj/pytorch-styleguide/issues/5))
  - In Windows, background generator could not be supported. So if error occurs, set false to `use_background_generator` in config
- Training state and network checkpoint saving, loading
    - Training state includes not only network weights, but also optimizer, step, epoch.
    - Checkpoint includes only network weights. This could be used for inference. 
- Config with yaml file / easy dot-style access to config
- Code lint / CI

## Setup

### Install requirements

- python3 (3.6, 3.7, 3.8 is tested)
- `pip install -r requirements.txt`
- proper PyTorch install (https://pytorch.org/get-started/)

### Config

- Config is written in yaml file(default: `config/default.yaml`)
- data field
    - Configs for Dataloader
    - glob `train_dir` / `test_dir` with `file_format` for Dataloader
- train/test field
    - Configs for training options
    - `random_seed` is for setting python, numpy, pytorch random seed
    - `num_iter` is for end iteration step of training.
    - `optimizer` is for selecting optimizer. Only `adam optimizer` is supported for now.
- model field
    - Configs for Network architecture and options for model like device
    - You can add configs in yaml format to config your network
- log field
    - Configs for logging include tensorboard / wandb logging
    - `name` is train name you run. 
    - `summary_interval` and `checkpoint_interval` are interval of step between training logging and checkpoint saving
    - checkpoint and logs (include tensorboard) are saved under `chkpt_dir/name` and `log_dir/name`.
- load field
    - loading from wandb server is supported
    - `wandb_load_path` is `Run path` in overview of run. If you don't want to use wandb load, this field should be `~`
    - `network_chkpt_path` is path to network checkpoint file.
    If using wandb loading, this field should be checkpoint file name of wandb run
    - `resume_state_path` is path to training state file.
    If using wandb loading, this field should be training state file name of wandb run 

### Code lint

1. `pip install -r requirements-dev.txt` for install develop dependencies (this requires python 3.6 and above because of black)

1. `pre-commit install` for adding pre-commit to git hook

## Train

- `python trainer.py -c config/path/to/file -n model_name`
