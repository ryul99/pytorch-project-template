import glob
import os
from enum import Enum, auto

import hydra
import torch
import torchvision
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class DataloaderMode(Enum):
    train = auto()
    test = auto()
    inference = auto()


class DataLoader_(DataLoader):
    # ref: https://github.com/IgorSusmelj/pytorch-styleguide/issues/5#issuecomment-495090086
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def create_dataloader(cfg, mode, rank):
    if cfg.data.use_background_generator:
        data_loader = DataLoader_
    else:
        data_loader = DataLoader
    dataset = Dataset_(cfg, mode)
    train_use_shuffle = True
    sampler = None
    if cfg.dist.gpus > 0 and cfg.data.divide_dataset_per_gpu:
        sampler = DistributedSampler(dataset, cfg.dist.gpus, rank)
        train_use_shuffle = False
    if mode is DataloaderMode.train:
        return data_loader(
            dataset=dataset,
            batch_size=cfg.train.batch_size,
            shuffle=train_use_shuffle,
            sampler=sampler,
            num_workers=cfg.train.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    elif mode is DataloaderMode.test:
        return data_loader(
            dataset=dataset,
            batch_size=cfg.test.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=cfg.test.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    else:
        raise ValueError(f"invalid dataloader mode {mode}")


class Dataset_(Dataset):
    def __init__(self, cfg, mode):
        self.cfg = cfg
        self.mode = mode
        if mode is DataloaderMode.train:
            # self.data_dir = self.cfg.data.train_dir
            # TODO: This is example code. You should change this part as you need
            self.dataset = torchvision.datasets.MNIST(
                root=hydra.utils.to_absolute_path("dataset/meta"),
                train=True,
                transform=torchvision.transforms.ToTensor(),
                download=False,
            )
        elif mode is DataloaderMode.test:
            # self.data_dir = self.cfg.data.test_dir
            # TODO: This is example code. You should change this part as you need
            self.dataset = torchvision.datasets.MNIST(
                root=hydra.utils.to_absolute_path("dataset/meta"),
                train=False,
                transform=torchvision.transforms.ToTensor(),
                download=False,
            )
        else:
            raise ValueError(f"invalid dataloader mode {mode}")
        # self.dataset_files = sorted(
        #     map(
        #         os.path.abspath,
        #         glob.glob(os.path.join(self.data_dir, self.cfg.data.file_format)),
        #     )
        # )
        # self.dataset = list()
        # for dataset_file in self.dataset_files:
        #     pass

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
