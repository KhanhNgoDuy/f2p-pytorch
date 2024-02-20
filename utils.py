import torch
from torch.utils.data import DataLoader
from torch import optim
import os
import shutil

from basel_2017 import Basel2017Dataset


def get_dataloader(cfg):
    print('Generating dataset...')
    dataloaders = []
    for stage in ['train', 'val']:
        cfg['trainer']['stage'] = stage
        ds = Basel2017Dataset(cfg)
        dataloader = DataLoader(
            dataset=ds,
            batch_size=cfg['trainer']['batch_size'],
            num_workers=cfg['trainer']['num_workers']
            )
        dataloaders.append(dataloader)
    return dataloaders


def get_optimizer(model, cfg):
    return optim.Adam(model.parameters(), lr=cfg['imitator']['lr'])


def remove_folder(dir):
    try:
        shutil.rmtree(dir)
    except FileNotFoundError:
        pass


if __name__ == '__main__':
    from pathlib import Path

    remove_folder(Path('result'))