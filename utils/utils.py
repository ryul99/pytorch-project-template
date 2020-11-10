import subprocess
import yaml
import random
import numpy as np
import torch
import torch.distributed as dist
from copy import deepcopy
from datetime import datetime


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_logging_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def get_timestamp():
    return datetime.now().strftime("%y%m%d-%H%M%S")


def get_commit_hash():
    message = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    return message.strip().decode("utf-8")
