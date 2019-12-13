import numpy as np
from tensorboardX import SummaryWriter


class Writer(SummaryWriter):
    def __init__(self, hp, logdir):
        super(Writer, self).__init__(logdir)
        self.hp = hp

    def train_logging(self, train_loss, step):
        self.add_scalar('train_loss', train_loss, step)

    def test_logging(self, test_loss, step):
        self.add_scalar('test_loss', test_loss, step)
