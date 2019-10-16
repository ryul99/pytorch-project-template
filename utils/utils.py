import numpy as np
import subprocess
import torch.nn.functional as F


def get_commit_hash():
    message = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    return message.strip().decode('utf-8')
