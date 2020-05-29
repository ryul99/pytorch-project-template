import torch
import torch.nn as nn
import torch.nn.functional as F


class Net_arch(nn.Module):
    # Network architecture
    def __init__(self, hp):
        super(Net_arch, self).__init__()
        self.hp = hp
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
