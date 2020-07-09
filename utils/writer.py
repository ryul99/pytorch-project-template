import numpy as np
from torch.utils.tensorboard import SummaryWriter
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

    def logging_with_step(self, value, step, logging_name):
        if self.hp.log.use_tensorboard:
            self.tensorboard.add_scalar(logging_name, value, step)
        if self.hp.log.use_wandb:
            wandb.log({logging_name: value}, step=step)
