from torch.utils.data import Dataset, DataLoader
import torch
import glob
import os
from enum import Enum, auto
from prefetch_generator import BackgroundGenerator


class DataloaderMode(Enum):
    train = auto()
    test = auto()
    inference = auto()


class DataLoader_(DataLoader):
    # ref: https://github.com/IgorSusmelj/pytorch-styleguide/issues/5#issuecomment-495090086
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def create_dataloader(hp, mode, rank, world_size):
    if hp.data.use_background_generator:
        data_loader = DataLoader_
    else:
        data_loader = DataLoader
    if mode is DataloaderMode.train:
        return data_loader(
            dataset=Dataset_(hp, mode, rank, world_size),
            batch_size=hp.train.batch_size,
            shuffle=True,
            num_workers=hp.train.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    elif mode is DataloaderMode.test:
        return data_loader(
            dataset=Dataset_(hp, mode, rank, world_size),
            batch_size=hp.test.batch_size,
            shuffle=False,
            num_workers=hp.test.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    else:
        raise ValueError(f"invalid dataloader mode {mode}")


class Dataset_(Dataset):
    def __init__(self, hp, mode, rank, world_size):
        self.hp = hp
        self.mode = mode
        if mode is DataloaderMode.train:
            self.data_dir = hp.data.train_dir
        elif mode is DataloaderMode.test:
            self.data_dir = hp.data.test_dir
        else:
            raise ValueError(f"invalid dataloader mode {mode}")
        self.dataset_files = sorted(
            map(
                os.path.abspath,
                glob.glob(os.path.join(self.data_dir, self.hp.data.file_format)),
            )
        )
        self.dataset = list()
        for dataset_file in self.dataset_files:
            # TODO: This is example code. You should change this part as you need
            pass
        # TODO: This is example code. You should change this part as you need
        self.dataset = [(torch.rand(10), torch.rand(1)) for _ in range(64)]

        if self.hp.data.divide_dataset_per_gpu:
            self.dataset.sort()
            if world_size != 0:
                if len(self.dataset) % world_size != 0:
                    raise ValueError("world_size should be factor of dataset size")
                self.dataset = self.dataset[
                    rank
                    * len(self.dataset)
                    // world_size : (rank + 1)
                    * len(self.dataset)
                    // world_size
                ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
