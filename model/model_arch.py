import torch
import torch.nn as nn
import torch.nn.functional as F


class Net_arch(nn.Module):
    # Network architecture
    def __init__(self, hp):
        super(Net_arch, self).__init__()
        self.hp = hp

        # TODO: This is example code. You should change this part as you need
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        # TODO: This is example code. You should change this part as you need
        x = self.fc1(x)
        x = self.fc2(x)
        return x
