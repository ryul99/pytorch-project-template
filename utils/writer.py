import numpy as np
import wandb
from torch.utils.tensorboard import SummaryWriter


class Writer(SummaryWriter):
    def __init__(self, cfg, logdir):
        self.cfg = cfg
        if cfg.log.use_tensorboard:
            self.tensorboard = SummaryWriter(logdir)
        if cfg.log.use_wandb:
            wandb_init_conf = cfg.log.wandb_init_conf
            wandb.init(config=cfg, **wandb_init_conf)

    def logging_with_step(self, value, step, logging_name):
        if self.cfg.log.use_tensorboard:
            self.tensorboard.add_scalar(logging_name, value, step)
        if self.cfg.log.use_wandb:
            wandb.log({logging_name: value}, step=step)
