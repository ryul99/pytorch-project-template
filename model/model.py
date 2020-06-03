import torch
import torch.nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
from collections import OrderedDict
import os.path as osp
import wandb

from utils.utils import DotDict


class Model:
    def __init__(self, hp, net_arch, loss_f):
        self.hp = hp
        self.device = hp.model.device
        self.net = net_arch.to(self.device)
        self.input = None
        self.GT = None
        self.step = 0
        self.epoch = -1

        # init optimizer
        if hp.train.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                self.net.parameters(), lr=hp.train.adam.initlr
            )
        else:
            raise Exception("%s optimizer not supported" % hp.train.optimizer)

        # init loss
        self.loss_f = loss_f
        self.log = DotDict()

    def feed_data(self, **data):  # data's keys: input, GT
        for k, v in data.items():
            data[k] = v.to(self.hp.model.device)
        self.input = data.get("input")
        self.GT = data.get("GT")

    def optimize_parameters(self):
        self.net.train()
        self.optimizer.zero_grad()
        output = self.net(self.input)
        loss_v = self.loss_f(self.GT, output)
        loss_v.backward()
        self.optimizer.step()

        # set log
        self.log.loss_v = loss_v.item()

    def inference(self):
        self.net.eval()
        output = self.net(self.input)
        return output

    def save_network(self, logger):
        save_filename = "%s_%d.pt" % (self.hp.log.name, self.step)
        save_path = osp.join(self.hp.log.chkpt_dir, save_filename)
        state_dict = self.net.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)
        if self.hp.log.use_wandb:
            torch.save(state_dict, osp.join(wandb.run.dir, save_filename))
            wandb.save(osp.join(wandb.run.dir, save_filename))
        logger.info("Saved network checkpoint to: %s" % save_path)

    def load_network(self, loaded_clean_net=None, logger=None):
        if loaded_clean_net is None:
            if self.hp.log.use_wandb is not None:
                self.hp.load.network_chkpt_path = wandb.restore(
                    self.hp.load.network_chkpt_path,
                    run_path=self.hp.load.wandb_load_path,
                ).name
            loaded_net = torch.load(self.hp.load.network_chkpt_path)
            loaded_clean_net = OrderedDict()  # remove unnecessary 'module.'
            for k, v in loaded_net.items():
                if k.startswith("module."):
                    loaded_clean_net[k[7:]] = v
                else:
                    loaded_clean_net[k] = v

        self.net.load_state_dict(loaded_clean_net, strict=self.hp.load.strict_load)
        if logger is not None:
            logger.info("Checkpoint %s is loaded" % self.hp.load.network_chkpt_path)

    def save_training_state(self, logger):
        save_filename = "%s_%d.state" % (self.hp.log.name, self.step)
        save_path = osp.join(self.hp.log.chkpt_dir, save_filename)
        state = {
            "model": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
        }
        torch.save(state, save_path)
        if self.hp.log.use_wandb:
            torch.save(state, osp.join(wandb.run.dir, save_filename))
            wandb.save(osp.join(wandb.run.dir, save_filename))
        logger.info("Saved training state to: %s" % save_path)

    def load_training_state(self, logger):
        if self.hp.log.use_wandb is not None:
            self.hp.load.resume_state_path = wandb.restore(
                self.hp.load.resume_state_path, run_path=self.hp.load.wandb_load_path
            ).name
        resume_state = torch.load(self.hp.load.resume_state_path)

        self.load_network(loaded_clean_net=resume_state["model"], logger=logger)
        self.optimizer.load_state_dict(resume_state["optimizer"])
        self.step = resume_state["step"]
        self.epoch = resume_state["epoch"]
        logger.info("Resuming from training state: %s" % self.hp.load.resume_state_path)
