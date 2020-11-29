import argparse
import datetime
import itertools
import os
import random
import traceback

import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

from dataset.dataloader import DataloaderMode, create_dataloader
from model.model import Model
from model.model_arch import Net_arch
from utils.test_model import test_model
from utils.train_model import train_model
from utils.utils import get_logger, is_logging_process, set_random_seed
from utils.writer import Writer


def setup(cfg, rank):
    os.environ["MASTER_ADDR"] = cfg.dist.master_addr
    os.environ["MASTER_PORT"] = cfg.dist.master_port
    timeout_sec = 1800
    if cfg.dist.timeout is not None:
        os.environ["NCCL_BLOCKING_WAIT"] = "1"
        timeout_sec = cfg.dist.timeout
    timeout = datetime.timedelta(seconds=timeout_sec)

    # initialize the process group
    dist.init_process_group(
        cfg.dist.mode,
        rank=rank,
        world_size=cfg.dist.gpus,
        timeout=timeout,
    )


def cleanup():
    dist.destroy_process_group()


def distributed_run(fn, cfg):
    mp.spawn(fn, args=(cfg,), nprocs=cfg.dist.gpus, join=True)


def train_loop(rank, cfg):
    logger = get_logger(cfg, os.path.basename(__file__))
    if cfg.device == "cuda" and cfg.dist.gpus != 0:
        cfg.device = rank
        # turn off background generator when distributed run is on
        cfg.data.use_background_generator = False
        setup(cfg, rank)
        torch.cuda.set_device(cfg.device)

    # setup writer
    if is_logging_process():
        # set log/checkpoint dir
        os.makedirs(cfg.log.chkpt_dir, exist_ok=True)
        # set writer (tensorboard / wandb)
        writer = Writer(cfg, "tensorboard")
        cfg_str = OmegaConf.to_yaml(cfg)
        logger.info("Config:\n" + cfg_str)
        if cfg.data.train_dir == "" or cfg.data.test_dir == "":
            logger.error("train or test data directory cannot be empty.")
            raise Exception("Please specify directories of data")
        logger.info("Set up train process")
        logger.info("BackgroundGenerator is turned off when Distributed running is on")

        # download MNIST dataset before making dataloader
        # TODO: This is example code. You should change this part as you need
        _ = torchvision.datasets.MNIST(
            root=hydra.utils.to_absolute_path("dataset/meta"),
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True,
        )
        _ = torchvision.datasets.MNIST(
            root=hydra.utils.to_absolute_path("dataset/meta"),
            train=False,
            transform=torchvision.transforms.ToTensor(),
            download=True,
        )
    # Sync dist processes (because of download MNIST Dataset)
    if cfg.dist.gpus != 0:
        dist.barrier()

    # make dataloader
    if is_logging_process():
        logger.info("Making train dataloader...")
    train_loader = create_dataloader(cfg, DataloaderMode.train, rank)
    if is_logging_process():
        logger.info("Making test dataloader...")
    test_loader = create_dataloader(cfg, DataloaderMode.test, rank)

    # init Model
    net_arch = Net_arch(cfg)
    loss_f = torch.nn.CrossEntropyLoss()
    model = Model(cfg, net_arch, loss_f, rank)

    # load training state / network checkpoint
    if cfg.load.resume_state_path is not None:
        model.load_training_state()
    elif cfg.load.network_chkpt_path is not None:
        model.load_network()
    else:
        if is_logging_process():
            logger.info("Starting new training run.")

    try:
        if cfg.dist.gpus == 0 or cfg.data.divide_dataset_per_gpu:
            epoch_step = 1
        else:
            epoch_step = cfg.dist.gpus
        for model.epoch in itertools.count(model.epoch + 1, epoch_step):
            if model.epoch > cfg.num_epoch:
                break
            train_model(cfg, model, train_loader, writer)
            if model.epoch % cfg.log.chkpt_interval == 0:
                model.save_network()
                model.save_training_state()
            test_model(cfg, model, test_loader, writer)
        if is_logging_process():
            logger.info("End of Train")
    except Exception as e:
        if is_logging_process():
            logger.error(traceback.format_exc())
        else:
            traceback.print_exc()
    finally:
        if cfg.dist.gpus != 0:
            cleanup()


@hydra.main(config_path="config", config_name="default")
def main(hydra_cfg):
    hydra_cfg.device = hydra_cfg.device.lower()
    with open_dict(hydra_cfg):
        hydra_cfg.job_logging_cfg = HydraConfig.get().job_logging
    # random seed
    if hydra_cfg.random_seed is None:
        hydra_cfg.random_seed = random.randint(1, 10000)
    set_random_seed(hydra_cfg.random_seed)

    if hydra_cfg.dist.gpus < 0:
        hydra_cfg.dist.gpus = torch.cuda.device_count()
    if hydra_cfg.device == "cpu" or hydra_cfg.dist.gpus == 0:
        hydra_cfg.dist.gpus = 0
        train_loop(0, hydra_cfg)
    else:
        distributed_run(train_loop, hydra_cfg)


if __name__ == "__main__":
    main()
