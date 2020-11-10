import subprocess
import random
import logging
import os.path as osp
import numpy as np
import torch
import torch.distributed as dist
from datetime import datetime
from omegaconf import OmegaConf


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_logging_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def get_logger(cfg, name=None, log_file_path=None):
    # log_file_path is used when unit testing
    if is_logging_process():
        project_root_path = osp.dirname(osp.dirname(osp.abspath(__file__)))
        hydra_conf = OmegaConf.load(osp.join(project_root_path, "config/default.yaml"))

        job_logging_name = None
        for job_logging_name in hydra_conf.defaults:
            if isinstance(job_logging_name, dict):
                job_logging_name = job_logging_name.get("hydra/job_logging")
                if job_logging_name is not None:
                    break
            job_logging_name = None
        if job_logging_name is None:
            job_logging_name = "custom"  # default name

        logging_conf = OmegaConf.load(
            osp.join(
                project_root_path,
                "config/hydra/job_logging",
                job_logging_name + ".yaml",
            )
        )
        if log_file_path is not None:
            logging_conf.handlers.file.filename = log_file_path
        logging.config.dictConfig(OmegaConf.to_container(logging_conf, resolve=True))
    return logging.getLogger(name)


def get_timestamp():
    return datetime.now().strftime("%y%m%d-%H%M%S")


def get_commit_hash():
    message = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    return message.strip().decode("utf-8")
