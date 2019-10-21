import numpy as np
from tensorboardX import SummaryWriter
from . import plotting as plt


class Writer(SummaryWriter):
    def __init__(self, hp, logdir):
        super(type(self), self).__init__(logdir)
        self.hp = hp

    def log_training(self, train_loss, step):
        self.add_scalar('train_loss', train_loss, step)

    def log_validation(self, test_loss, step):
        self.add_scalar('test_loss', test_loss, step)
