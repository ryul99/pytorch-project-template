import os
import logging
from utils.utils import get_timestamp


def make_logger(cfg):
    # set log/checkpoint dir
    timestamp = get_timestamp()
    cfg.log.chkpt_dir = os.path.join(
        cfg.log.chkpt_dir, "%s-%s" % (cfg.log.name, timestamp)
    )
    cfg.log.log_dir = os.path.join(cfg.log.log_dir, cfg.log.name)
    os.makedirs(cfg.log.chkpt_dir, exist_ok=True)
    os.makedirs(cfg.log.log_dir, exist_ok=True)
    cfg.log.log_file_path = os.path.join(
        cfg.log.log_dir, "%s-%s.log" % (cfg.log.name, timestamp)
    )

    # set logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(cfg.log.log_file_path),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger()
    return logger
