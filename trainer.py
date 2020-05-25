import os
import time
import logging
import argparse
import yaml

from utils.train_model import train
from utils.hparams import load_hparam
from utils.writer import Writer
from dataset.dataloader import create_dataloader, DataloaderMode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for config.")
    parser.add_argument('-p', '--checkpoint_path', type=str, default=None,
                        help="path of checkpoint pt file for resuming")
    parser.add_argument('-n', '--name', type=str, required=True,
                        help="Name of the model. Used for both logging and saving chkpt.")
    args = parser.parse_args()
    hp = load_hparam(args.config)
    hp_str = yaml.dump(hp.to_dict())
    args_str = yaml.dump(vars(args))

    pt_dir = os.path.join(hp.log.chkpt_dir, args.name)
    log_dir = os.path.join(hp.log.log_dir, args.name)
    os.makedirs(pt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir,
                                             '%s-%d.log' % (args.name, time.time()))),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    logger.info('Config by yaml file')
    logger.info(hp_str)
    logger.info('Command Line Config')
    logger.info(args_str)

    if hp.data.train == '' or hp.data.test == '':
        logger.error("train or test data directory cannot be empty.")
        raise Exception("Please specify directories of data in %s" % args.config)

    writer = Writer(hp, log_dir)
    train_loader = create_dataloader(hp, args, DataloaderMode.train)
    test_loader = create_dataloader(hp, args, DataloaderMode.test)

    train(args, pt_dir, train_loader, test_loader, writer, logger, hp, hp_str)


if __name__ == '__main__':
    main()
