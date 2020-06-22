import subprocess
import yaml
import random
import numpy as np
import torch
from copy import deepcopy
from datetime import datetime


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_timestamp():
    return datetime.now().strftime("%y%m%d-%H%M%S")


def get_commit_hash():
    message = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    return message.strip().decode("utf-8")


def load_hparam(filename):
    stream = open(filename, "r")
    docs = yaml.load_all(stream, Loader=yaml.Loader)
    hparam_dict = DotDict()
    for doc in docs:
        for k, v in doc.items():
            hparam_dict[k] = DotDict(v)
    return hparam_dict


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dict_=None):
        super().__init__()
        if dict_ is not None:
            if not isinstance(dict_, dict):
                raise ValueError
            for k, v in dict_.items():
                if isinstance(v, dict):
                    self[k] = DotDict(v)
                else:
                    self[k] = v

    def __copy__(self):
        copy = type(self)()
        for k, v in self.items():
            copy[k] = v
        return copy

    def __deepcopy__(self, memodict={}):
        copy = type(self)()
        memodict[id(self)] = copy
        for k, v in self.items():
            copy[k] = deepcopy(v, memodict)
        return copy

    def __getstate__(self):
        return self.to_dict()

    def __setstate__(self, state):
        self.__init__(state)

    def to_dict(self):
        output_dict = dict()
        for k, v in self.items():
            if isinstance(v, DotDict):
                output_dict[k] = v.to_dict()
            else:
                output_dict[k] = v
        return output_dict
