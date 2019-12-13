import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, hp):
        super(Net, self).__init__()
        self.hp = hp
        raise NotImplementedError
    
    def forward(self, x):
        raise NotImplementedError

    def get_loss(self, output, target):
        raise NotImplementedError
