from torch.utils.data import Dataset, DataLoader


def create_dataloader(hp, args, train):
    if train:
        return DataLoader(dataset=Dataset_(hp, args, True),
                          batch_size=hp.train.batch_size,
                          shuffle=True,
                          num_workers=hp.train.num_workers,
                          pin_memory=True,
                          drop_last=True)
    else:
        return DataLoader(dataset=Dataset_(hp, args, False),
                          batch_size=hp.test.batch_size,
                          shuffle=False,
                          num_workers=hp.test.num_workers,
                          pin_memory=True,
                          drop_last=True)


class Dataset_(Dataset):
    def __init__(self, hp, args, train):
        self.hp = hp
        self.args = args
        self.train = train
        self.data_dir = hp.data.train if train else hp.data.test
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
