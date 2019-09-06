import numpy as np
from tensorboardX import SummaryWriter
from . import plotting as plt


class Writer(SummaryWriter):
    def __init__(self, hp, logdir):
        super(type(self), self).__init__(logdir)
        self.hp = hp

    def log_training(self, train_loss, step):
        self.add_scalar('train_loss', train_loss, step)

    def log_validation(self, fpr, tpr, thresholds, eer, thr, step):
        far = fpr
        frr = 1.0 - tpr
        self.add_scalar('eer', eer, step)
        self.add_scalar('threshold', thr, step)
        self.add_image('FAR-FRR',
                       plt.plot_far_frr_to_numpy(thresholds, far, frr), step)
        self.add_image('ROC',
                       plt.plot_roc_to_numpy(far, tpr), step)
