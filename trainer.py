import datetime
import argparse
import yaml
import itertools
import traceback
import random
import os
import logging

import hydra
import torch
import torchvision
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from model.model_arch import Net_arch
from model.model import Model
from utils.train_model import train_model
from utils.test_model import test_model
from utils.utils import set_random_seed
from utils.writer import Writer
from dataset.dataloader import create_dataloader, DataloaderMode


logger = logging.getLogger(os.path.basename(__file__))


def setup(cfg, rank):
    os.environ["MASTER_ADDR"] = cfg.train.train.dist.master_addr
    os.environ["MASTER_PORT"] = cfg.train.train.dist.master_port
    timeout_sec = 1800
    if cfg.train.train.dist.timeout is not None:
        os.environ["NCCL_BLOCKING_WAIT"] = "1"
        timeout_sec = cfg.train.train.dist.timeout
    timeout = datetime.timedelta(seconds=timeout_sec)

    # initialize the process group
    dist.init_process_group(
        cfg.train.train.dist.mode,
        rank=rank,
        world_size=cfg.train.train.dist.gpus,
        timeout=timeout,
    )


def cleanup():
    dist.destroy_process_group()


def distributed_run(fn, cfg):
    mp.spawn(fn, args=(cfg,), nprocs=cfg.train.train.dist.gpus, join=True)


def train_loop(rank, cfg):
    if cfg.train.model.device == "cuda" and cfg.train.train.dist.gpus != 0:
        cfg.train.model.device = rank
        # turn off background generator when distributed run is on
        cfg.train.data.use_background_generator = False
        setup(cfg, rank)
        torch.cuda.set_device(cfg.train.model.device)

    # setup writer
    if rank == 0:
        # set log/checkpoint dir
        os.makedirs(cfg.train.log.chkpt_dir, exist_ok=True)
        # set writer (tensorboard / wandb)
        writer = Writer(cfg, "tensorboard")
        cfg_str = OmegaConf.to_yaml(cfg)
        logger.info("Config:\n" + cfg_str)
        if cfg.train.data.train_dir == "" or cfg.train.data.test_dir == "":
            logger.error("train or test data directory cannot be empty.")
            raise Exception("Please specify directories of data")
        logger.info("Set up train process")
        logger.info("BackgroundGenerator is turned off when Distributed running is on")

        # download MNIST dataset before making dataloader
        # TODO: This is example code. You should change this part as you need
        _ = torchvision.datasets.MNIST(
            root="dataset/meta",
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True,
        )
        _ = torchvision.datasets.MNIST(
            root="dataset/meta",
            train=False,
            transform=torchvision.transforms.ToTensor(),
            download=True,
        )
    # Sync dist processes (because of download MNIST Dataset)
    if cfg.train.train.dist.gpus != 0:
        dist.barrier()

    # make dataloader
    if rank == 0:
        logger.info("Making train dataloader...")
    train_loader = create_dataloader(cfg, DataloaderMode.train, rank)
    if rank == 0:
        logger.info("Making test dataloader...")
    test_loader = create_dataloader(cfg, DataloaderMode.test, rank)

    # init Model
    net_arch = Net_arch(cfg)
    loss_f = torch.nn.CrossEntropyLoss()
    model = Model(cfg, net_arch, loss_f, rank)

    # load training state / network checkpoint
    if cfg.train.load.resume_state_path is not None:
        model.load_training_state(rank)
    elif cfg.train.load.network_chkpt_path is not None:
        model.load_network(rank=rank)
    else:
        if rank == 0:
            logger.info("Starting new training run.")

    try:
        if cfg.train.train.dist.gpus == 0 or cfg.train.data.divide_dataset_per_gpu:
            epoch_step = 1
        else:
            epoch_step = cfg.train.train.dist.gpus
        for model.epoch in itertools.count(model.epoch + 1, epoch_step):
            if model.epoch > cfg.train.train.num_epoch:
                break
            train_model(cfg, model, train_loader, writer, rank)
            if model.epoch % cfg.train.log.chkpt_interval == 0:
                model.save_network(rank)
                model.save_training_state(rank)
            test_model(cfg, model, test_loader, writer, rank)
        if rank == 0:
            logger.info("End of Train")
    except Exception as e:
        if rank == 0:
            logger.error(traceback.format_exc())
        else:
            traceback.print_exc()
    finally:
        if cfg.train.train.dist.gpus != 0:
            cleanup()


@hydra.main(config_path="config/train.yaml")
def main(hydra_cfg):
    hydra_cfg.train.model.device = hydra_cfg.train.model.device.lower()

    # random seed
    if hydra_cfg.train.train.random_seed is None:
        hydra_cfg.train.train.random_seed = random.randint(1, 10000)
    set_random_seed(hydra_cfg.train.train.random_seed)

    if hydra_cfg.train.train.dist.gpus < 0:
        hydra_cfg.train.train.dist.gpus = torch.cuda.device_count()
    if hydra_cfg.train.model.device == "cpu" or hydra_cfg.train.train.dist.gpus == 0:
        hydra_cfg.train.train.dist.gpus = 0
        train_loop(0, hydra_cfg)
    else:
        distributed_run(train_loop, hydra_cfg)


if __name__ == "__main__":
    main()
