<div align="center">
    <img src="assets/icon.png"/>
    <h1><code>
        Pytorch Project Template
    </h1></code>
    <p>
        <img src="https://img.shields.io/github/license/ryul99/pytorch-project-template"/>
        <a href="https://pycqa.github.io/isort/"><img src="https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336"/></a>
    </p>
</div>

## Feature

- TensorBoard / [wandb](https://www.wandb.com/) support
- Background generator is used ([reason of using background generator](https://github.com/IgorSusmelj/pytorch-styleguide/issues/5))
  - In Windows, background generator could not be supported. So if error occurs, set false to `use_background_generator` in config
- Training state and network checkpoint saving, loading
    - Training state includes not only network weights, but also optimizer, step, epoch.
    - Checkpoint includes only network weights. This could be used for inference. 
- [Hydra](https://hydra.cc) and [Omegaconf](https://github.com/omry/omegaconf) is supported
- Distributed Learning using Distributed Data Parallel is supported
- Config with yaml file / easy dot-style access to config
- Code lint / CI
- Code Testing with pytest

## Code Structure

- `assets` dir: icon image of `Pytorch Project Template`. You can remove this directory.
- `config` dir: directory for config files
- `dataset` dir: dataloader and dataset codes are here. Also, put dataset in `meta` dir.
- `model` dir: `model.py` is for wrapping network architecture. `model_arch.py` is for coding network architecture.
- `tests` dir: directory for `pytest` testing codes. You can check your network's flow of tensor by fixing `tests/model/net_arch_test.py`. 
Just copy & paste `Net_arch.forward` method to  `net_arch_test.py` and add `assert` phrase to check tensor.
- `utils` dir:
    - `train_model.py` and `test_model.py` are for train and test model once.
    - `utils.py` is for utility. random seed setting, dot-access hyper parameter, get commit hash, etc are here. 
    - `writer.py` is for writing logs in tensorboard / wandb.
- `trainer.py` file: this is for setting up and iterating epoch.

## Setup

### Install requirements

- python3 (3.6, 3.7, 3.8 is tested)
- Write PyTorch version which you want to `requirements.txt`. (https://pytorch.org/get-started/)
- `pip install -r requirements.txt`

### Config

- Config is written in yaml file
    - You can choose configs at `config/default.yaml`. Custom configs are under `config/job/`
- `name` is train name you run.
- `working_dir` is root directory for saving checkpoints, logging logs.
- `device` is device mode for running your model. You can choose `cpu` or `cuda`
- `data` field
    - Configs for Dataloader.
    - glob `train_dir` / `test_dir` with `file_format` for Dataloader.
    - If `divide_dataset_per_gpu` is true, origin dataset is divide into sub dataset for each gpu. 
    This could mean the size of origin dataset should be multiple of number of using gpu.
    If this option is false, dataset is not divided but epoch goes up in multiple of number of gpus.
- `train`/`test` field
    - Configs for training options.
    - `random_seed` is for setting python, numpy, pytorch random seed.
    - `num_epoch` is for end iteration step of training.
    - `optimizer` is for selecting optimizer. Only `adam optimizer` is supported for now.
    - `dist` is for configuring Distributed Data Parallel.
        - `gpus` is the number that you want to use with DDP (`gpus` value is used at `world_size` in DDP).
        Not using DDP when `gpus` is 0, using all gpus when `gpus` is -1.
        - `timeout` is seconds for timeout of process interaction in DDP.
        When this is set as `~`, default timeout (1800 seconds) is applied in `gloo` mode and timeout is turned off in `nccl` mode.
- `model` field
    - Configs for Network architecture and options for model.
    - You can add configs in yaml format to config your network.
- `log` field
    - Configs for logging include tensorboard / wandb logging. 
    - `summary_interval` and `checkpoint_interval` are interval of step and epoch between training logging and checkpoint saving.
    - checkpoint and logs are saved under `working_dir/chkpt_dir` and `working_dir/trainer.log`. Tensorboard logs are saving under `working_dir/outputs/tensorboard`
- `load` field
    - loading from wandb server is supported
    - `wandb_load_path` is `Run path` in overview of run. If you don't want to use wandb load, this field should be `~`.
    - `network_chkpt_path` is path to network checkpoint file.
    If using wandb loading, this field should be checkpoint file name of wandb run.
    - `resume_state_path` is path to training state file.
    If using wandb loading, this field should be training state file name of wandb run.

### Code lint

1. `pip install -r requirements-dev.txt` for install develop dependencies (this requires python 3.6 and above because of black)

1. `pre-commit install` for adding pre-commit to git hook

## Train

- `python trainer.py working_dir=$(pwd)`

## Inspired by

- https://github.com/open-mmlab/mmsr
- https://github.com/allenai/allennlp (test case writing)
