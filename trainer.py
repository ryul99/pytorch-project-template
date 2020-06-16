import argparse
import yaml
import itertools
import traceback
import random
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from model.model_arch import Net_arch
from model.model import Model
from utils.train_model import train_model
from utils.test_model import test_model
from utils.utils import load_hparam, set_random_seed, DotDict
from utils.writer import Writer
from utils.logger import make_logger
from dataset.dataloader import create_dataloader, DataloaderMode


def setup(hp, rank, world_size):
    os.environ["MASTER_ADDR"] = hp.train.dist.master_addr
    os.environ["MASTER_PORT"] = hp.train.dist.master_port

    # initialize the process group
    dist.init_process_group(hp.train.dist.mode, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def distributed_run(fn, hp, world_size):
    mp.spawn(fn, args=(hp, world_size,), nprocs=world_size, join=True)


def train_loop(rank, hp, world_size=0):
    # reload hp
    hp = DotDict(hp)

    if hp.model.device == "cuda" and world_size != 0:
        hp.model.device = rank
        setup(hp, rank, world_size)
        torch.cuda.set_device(hp.model.device)

    # setup logger / writer
    if rank != 0:
        logger = None
        writer = None
    else:
        # set logger
        logger = make_logger(hp)
        # set writer (tensorboard / wandb)
        writer = Writer(hp, hp.log.log_dir)
        hp_str = yaml.dump(hp.to_dict())
        logger.info("Config:")
        logger.info(hp_str)
        if hp.data.train_dir == "" or hp.data.test_dir == "":
            logger.error("train or test data directory cannot be empty.")
            raise Exception("Please specify directories of data")
        logger.info("Set up train process")

    # make dataloader
    if logger is not None:
        logger.info("Making train dataloader...")
    train_loader = create_dataloader(hp, DataloaderMode.train, rank, world_size)
    if logger is not None:
        logger.info("Making test dataloader...")
    test_loader = create_dataloader(hp, DataloaderMode.test, rank, world_size)

    # init Model
    net_arch = Net_arch(hp)
    loss_f = torch.nn.MSELoss()
    model = Model(hp, net_arch, loss_f, rank, world_size)

    # load training state / network checkpoint
    if hp.load.resume_state_path is not None:
        model.load_training_state(logger)
    elif hp.load.network_chkpt_path is not None:
        model.load_network(logger=logger)
    else:
        if logger is not None:
            logger.info("Starting new training run.")

    try:
        epoch_step = 1 if hp.data.divide_dataset_per_gpu else world_size
        for model.epoch in itertools.count(model.epoch + 1, epoch_step):
            if model.epoch > hp.train.num_iter:
                break
            train_model(hp, model, train_loader, writer, logger)
            if model.epoch % hp.log.chkpt_interval == 0:
                model.save_network(logger)
                model.save_training_state(logger)
            test_model(hp, model, test_loader, writer)
        if logger is not None:
            logger.info("End of Train")
    except Exception as e:
        if logger is not None:
            logger.info("Exiting due to exception: %s" % e)
        traceback.print_exc()
    finally:
        if world_size != 0:
            cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="yaml file for config."
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default=None,
        help="Name of the model. Used for both logging and saving chkpt.",
    )
    args = parser.parse_args()
    hp = load_hparam(args.config)
    hp.model.device = hp.model.device.lower()

    if args.name is not None:
        hp.log.name = args.name

    # random seed
    if hp.train.random_seed is None:
        hp.train.random_seed = random.randint(1, 10000)
    set_random_seed(hp.train.random_seed)

    if hp.train.dist.gpus < 0:
        hp.train.dist.gpus = torch.cuda.device_count()
    if hp.model.device == "cpu" or hp.train.dist.gpus == 0:
        train_loop(0, hp)
    else:
        distributed_run(train_loop, hp.to_dict(), hp.train.dist.gpus)


if __name__ == "__main__":
    main()
