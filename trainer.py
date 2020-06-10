import argparse
import yaml
import itertools
import traceback
import torch
import random

from model.model_arch import Net_arch
from model.model import Model
from utils.train_model import train_model
from utils.test_model import test_model
from utils.utils import load_hparam, set_random_seed
from utils.writer import Writer
from utils.logger import make_logger
from dataset.dataloader import create_dataloader, DataloaderMode


def train_loop(hp, logger, writer):
    # make dataloader
    logger.info("Making train dataloader...")
    train_loader = create_dataloader(hp, DataloaderMode.train)
    logger.info("Making test dataloader...")
    test_loader = create_dataloader(hp, DataloaderMode.test)

    # init Model
    net_arch = Net_arch(hp)
    loss_f = torch.nn.MSELoss()
    model = Model(hp, net_arch, loss_f)

    if hp.load.resume_state_path is not None:
        model.load_training_state(logger)
    else:
        logger.info("Starting new training run.")

    try:
        for model.epoch in itertools.count(model.epoch + 1):
            if model.epoch > hp.train.num_iter:
                break
            train_model(hp, model, train_loader, writer, logger)
            if model.epoch % hp.log.chkpt_interval == 0:
                model.save_network(logger)
                model.save_training_state(logger)
            test_model(hp, model, test_loader, writer)
        logger.info("End of Train")
    except Exception as e:
        logger.info("Exiting due to exception: %s" % e)
        traceback.print_exc()


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

    if args.name is not None:
        hp.log.name = args.name

    # random seed
    if hp.train.random_seed is None:
        hp.train.random_seed = random.randint(1, 10000)
    set_random_seed(hp.train.random_seed)

    # set logger
    logger = make_logger(hp)

    # set writer (tensorboard / wandb)
    writer = Writer(hp, hp.log.log_dir)

    hp_str = yaml.dump(hp.to_dict())
    logger.info("Config:")
    logger.info(hp_str)

    if hp.data.train_dir == "" or hp.data.test_dir == "":
        logger.error("train or test data directory cannot be empty.")
        raise Exception("Please specify directories of data in %s" % args.config)

    train_loop(hp, logger, writer)


if __name__ == "__main__":
    main()
