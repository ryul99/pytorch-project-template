# modified from https://github.com/HarryVolek/PyTorch_Speaker_Verification

import os
import yaml


def load_hparam_str(hp_str):
    path = os.path.join('config', 'temp-restore.yaml')
    with open(path, 'w') as f:
        f.write(hp_str)
    return HParam(path)


def load_hparam(filename):
    stream = open(filename, 'r')
    docs = yaml.load_all(stream, Loader=yaml.Loader)
    hparam_dict = dict()
    for doc in docs:
        for k, v in doc.items():
            hparam_dict[k] = v
    return hparam_dict


def merge_dict(user, default):
    if isinstance(user, dict) and isinstance(default, dict):
        for k, v in default.items():
            if k not in user:
                user[k] = v
            else:
                user[k] = merge_dict(user[k], v)
    return user


class DotDict(dict):
    """
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct=None):
        dct = dict() if not dct else dct
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value


class HParam(DotDict):

    def __init__(self, file):
        super(DotDict, self).__init__()
        hp_dict = load_hparam(file)
        hp_dotdict = DotDict(hp_dict)
        for k, v in hp_dotdict.items():
            setattr(self, k, v)
            
    __getattr__ = DotDict.__getitem__
    __setattr__ = DotDict.__setitem__
    __delattr__ = DotDict.__delitem__
