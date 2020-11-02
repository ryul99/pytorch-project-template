import torch
import torch.nn as nn
import torch.nn.functional as F


class Net_arch(nn.Module):
    # Network architecture
    def __init__(self, cfg):
        super(Net_arch, self).__init__()
        self.cfg = cfg

        # TODO: This is example code. You should change this part as you need
        self.lrelu = nn.LeakyReLU()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 4, 3, 2, 1), self.lrelu)
        self.conv2 = nn.Sequential(nn.Conv2d(4, 4, 3, 2, 1), self.lrelu)
        self.fc = nn.Linear(7 * 7 * 4, 10)

    def forward(self, x):  # x: (B,1,28,28)
        # TODO: This is example code. You should change this part as you need
        x = self.conv1(x)  # x: (B,4,14,14)
        x = self.conv2(x)  # x: (B,4,7,7)
        x = torch.flatten(x, 1)  # x: (B,4*7*7)
        x = self.fc(x)  # x: (B,10)
        return x
