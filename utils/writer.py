import numpy as np
from tensorboardX import SummaryWriter


class Writer(SummaryWriter):
    def __init__(self, hp, logdir):
        super(type(self), self).__init__(logdir)
        self.hp = hp

    def training_logging(self, train_loss, step):
        self.add_scalar('train_loss', train_loss, step)

    def validation_logging(self, test_loss, step):
        self.add_scalar('test_loss', test_loss, step)
