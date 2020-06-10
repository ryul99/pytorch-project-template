import os
import logging
import time


def make_logger(hp):
    # set log/checkpoint dir
    hp.log.chkpt_dir = os.path.join(hp.log.chkpt_dir, hp.log.name)
    hp.log.log_dir = os.path.join(hp.log.log_dir, hp.log.name)
    os.makedirs(hp.log.chkpt_dir, exist_ok=True)
    os.makedirs(hp.log.log_dir, exist_ok=True)

    # set logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(hp.log.log_dir, "%s-%d.log" % (hp.log.name, time.time()))
            ),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger()
    return logger
