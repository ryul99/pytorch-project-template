import numpy as np
from tensorboardX import SummaryWriter
import wandb


class Writer(SummaryWriter):
    def __init__(self, hp, logdir):
        self.hp = hp
        if hp.log.use_tensorboard:
            self.tensorboard = SummaryWriter(logdir)
        if hp.log.use_wandb:
            wandb_init_conf = hp.log.wandb_init_conf.to_dict()
            wandb_init_conf["config"] = hp.to_dict()
            wandb.init(**wandb_init_conf)

    def train_logging(self, train_loss, step):
        if self.hp.log.use_tensorboard:
            self.tensorboard.add_scalar("train_loss", train_loss, step)
        if self.hp.log.use_wandb:
            wandb.log({"train_loss": train_loss}, step=step)

    def test_logging(self, test_loss, step):
        if self.hp.log.use_tensorboard:
            self.tensorboard.add_scalar("test_loss", test_loss, step)
        if self.hp.log.use_wandb:
            wandb.log({"test_loss": test_loss}, step=step)
