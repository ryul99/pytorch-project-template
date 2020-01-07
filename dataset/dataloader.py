from torch.utils.data import Dataset, DataLoader
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


def create_dataloader(hp, args, mode):
    if mode is DataloaderMode.train:
        return DataLoader_(dataset=Dataset_(hp, args, mode),
                          batch_size=hp.train.batch_size,
                          shuffle=True,
                          num_workers=hp.train.num_workers,
                          pin_memory=True,
                          drop_last=True)
    elif mode is DataloaderMode.test:
        return DataLoader_(dataset=Dataset_(hp, args, mode),
                          batch_size=hp.test.batch_size,
                          shuffle=False,
                          num_workers=hp.test.num_workers,
                          pin_memory=True,
                          drop_last=True)
    else:
        raise ValueError(f'invalid dataloader mode {mode}')


class Dataset_(Dataset):
    def __init__(self, hp, args, mode):
        self.hp = hp
        self.args = args
        self.mode = mode
        if mode is DataloaderMode.train:
            self.data_dir = hp.data.train
        elif mode is DataloaderMode.test:
            self.data_dir = hp.data.test
        else:
            raise ValueError(f'invalid dataloader mode {mode}')
        self.dataset_files = sorted(map(os.path.abspath, glob.glob(os.path.join(self.data_dir, self.hp.data.file_format))))
        self.dataset = list()
        for dataset_file in self.dataset_files:
            pass
        raise NotImplementedError

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
