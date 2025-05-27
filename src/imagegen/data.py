import os
from typing import List

import numpy as np
from sklearn.datasets import load_digits
from torch.utils.data import Dataset, DataLoader

import random
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from imagegen.setup import ImgRunTracker


class Digits(Dataset):
    """Scikit-Learn Digits dataset."""

    def __init__(self, mode='train', transforms=None):
        digits = load_digits()
        if mode == 'train':
            self.data = digits.data[:1000].astype(np.float32)
        elif mode == 'val':
            self.data = digits.data[1000:1350].astype(np.float32)
        else:
            self.data = digits.data[1350:].astype(np.float32)

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transforms:
            sample = self.transforms(sample)
        return sample


def get_files(directories: List[str], max_mb: float) -> List[str]:
    all_files = []
    for directory in directories:
        for root, _, files in os.walk(directory):
            all_files.extend([os.path.join(root, f) for f in files])
    random.shuffle(all_files)
    cumsum_mb = np.cumsum([os.path.getsize(fname) / 1e6 for fname in all_files])
    use_files = all_files[0:np.sum(cumsum_mb <= max_mb)]
    if len(use_files) < len(all_files):
        print(f"Discarding {len(all_files) - len(use_files)} to stay under {max_mb} MB data")
    return use_files


class CroppedImageDataset(Dataset):
    def __init__(self, files, crop_size=240):
        self.files = files
        self.crop_size = crop_size
        self.transform = T.Compose([
            T.RandomCrop(crop_size),          # data augmentation
            T.ToTensor(),                     # (0,1)
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img)


class ImageDataset(Dataset):
    def __init__(self, files):
        self.files = files
        self.transform = T.Compose([
            T.ToTensor(),                     # (0,1)
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img)


def get_digits_ds():
    digits = Digits()
    return DataLoader(dataset=digits, batch_size=5, shuffle=True)


def get_celeb_ds():
    files = get_files(directories=["/Users/davidschneider/data/image/celeba/img_align_celeba_sample"], max_mb=100)    
    ds = ImageDataset(files=files)
    return DataLoader(dataset=ds, batch_size=5, shuffle=True)

    
def get_datasets(res: ImgRunTracker) -> DataLoader:
    files = get_files(**res.cfg['data'])
    train_n = min(len(files), int(round(len(files) * res.cfg['dataset']['train_ratio'])))
    train_files = files[0:train_n]
    val_files = files[train_n:]
    crop_size = res.cfg['dataset']['crop_size']
    return dict(train=CroppedImageDataset(files=train_files, crop_size=crop_size),
                val=CroppedImageDataset(files=val_files, crop_size=crop_size))
