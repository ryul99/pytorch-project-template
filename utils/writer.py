import numpy as np
from torch.utils.tensorboard import SummaryWriter
import wandb


class Writer(SummaryWriter):
    def __init__(self, cfg, logdir):
        self.cfg = cfg
        if cfg.train.log.use_tensorboard:
            self.tensorboard = SummaryWriter(logdir)
        if cfg.train.log.use_wandb:
            wandb_init_conf = cfg.train.log.wandb_init_conf
            wandb.init(config=cfg, **wandb_init_conf)

    def logging_with_step(self, value, step, logging_name):
        if self.cfg.train.log.use_tensorboard:
            self.tensorboard.add_scalar(logging_name, value, step)
        if self.cfg.train.log.use_wandb:
            wandb.log({logging_name: value}, step=step)
