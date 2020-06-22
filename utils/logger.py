import os
import logging
from utils.utils import get_timestamp


def make_logger(hp):
    # set log/checkpoint dir
    hp.log.chkpt_dir = os.path.join(hp.log.chkpt_dir, hp.log.name)
    hp.log.log_dir = os.path.join(hp.log.log_dir, hp.log.name)
    os.makedirs(hp.log.chkpt_dir, exist_ok=True)
    os.makedirs(hp.log.log_dir, exist_ok=True)

    hp.log.log_file_path = os.path.join(
        hp.log.log_dir, "%s-%s.log" % (hp.log.name, get_timestamp())
    )

    # set logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(hp.log.log_file_path), logging.StreamHandler(),],
    )
    logger = logging.getLogger()
    return logger
