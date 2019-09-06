import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, hp):
        super(type(self), self).__init__()
        self.hp = hp
        self.window = torch.hann_window(window_length=hp.audio.win_length)
        # model should be implemented

    def forward(self, x):
        raise NotImplementedError

    def get_loss(self, output, target):
        raise NotImplementedError
