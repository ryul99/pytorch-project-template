import yaml


def load_hparam(filename):
    stream = open(filename, 'r')
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

    def to_dict(self):
        output_dict = dict()
        for k, v in self.items():
            if isinstance(v, DotDict):
                output_dict[k] = v.to_dict()
            else:
                output_dict[k] = v
        return output_dict
