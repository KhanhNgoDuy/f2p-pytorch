import cv2
import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path

from transform import get_transform_compose


def load_dataset(path, start, end, transform):
    """
    Provide the path to the `output` folder
    of `parametric-face-image-generator`
    """
    dataset = []
    img_folder = path / 'img'
    param_folder = path / 'csv'
    
    for i in range(start, end):
        i = str(i)

        # Process data
        img = img_folder / i / f"{i}_1.png"
        img = cv2.imread(img.as_posix())
        if transform:
            img = transform(img)
        param = pd.read_csv((param_folder / i / f"{i}_1.csv").as_posix(), header=None).values
        param = torch.tensor(param).squeeze(0)
        
        dataset.append([img, param])

    return dataset


class Basel2017Dataset(Dataset):
    def __init__(self, cfg):
        cfg = cfg['trainer']

        self.data_path = Path(cfg['path']).expanduser().resolve()
        self.stage = cfg['stage']
        self.transform = get_transform_compose() if cfg['transform'] else None
        self.split = {
            'train': 0.4,
            'val': 0.3,
            'test': 0.3
        }
        self.prepare_data()
    
    def __getitem__(self, index):
        """
        Return [img, param]
        """
        return self.data[index]

    def prepare_data(self):
        train_idx, val_idx, test_idx = self.get_split_index()

        if self.stage == 'train':
            self.data = load_dataset(self.data_path, start=0, end=train_idx, transform=self.transform)
        elif self.stage == 'val':
            self.data = load_dataset(self.data_path, start=train_idx, end=val_idx, transform=self.transform)
        elif self.stage == 'test':
            self.data = load_dataset(self.data_path, start=val_idx, end= test_idx, transform=self.transform)
        else:
            raise AssertionError('Stage must be in ["train", "val", "test"]')

    def get_split_index(self):
        n_full_instance = len(list((self.data_path / 'img').iterdir()))
        train_idx = round(n_full_instance * self.split['train'])
        val_idx = round(n_full_instance * (self.split['train'] + self.split['val']))
        test_idx = n_full_instance

        return train_idx, val_idx, test_idx
    
    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    import yaml

    with open('/home/khanh/content/projects/implement-F2P (pytorch)/config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
        cfg_dataset = cfg['trainer']
    dataset = Basel2017Dataset(cfg)
    img, param = dataset[0]
    print(f'=== {img.shape}\t{param.shape}')
