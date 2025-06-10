import os
import shutil
from tqdm import tqdm
from typing import List, Tuple

import numpy as np
from sklearn.datasets import load_digits
from torch.utils.data import Dataset, DataLoader

import random
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import tqdm
from imagegen.setup import ImgRunTracker


def get_num_channels(mode: str) -> int:
    channels = {
        "1": 1,       # (grayscale)
        "L": 1,       # (grayscale)
        "RGB": 3,
        "RGBA": 4,
        "CMYK": 4,
        "YCbCr": 3,
    }

    if mode not in channels:
        raise ValueError(f"Mode {mode} not known. Known modes are {channels.keys()}")
    
    return channels[mode]


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
    def __init__(self, files, zero_centered=False):
        self.files = files
        self.zero_centered = zero_centered
        if zero_centered:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Lambda(lambda x: (x - 0.5) * 2),
            ])
        else:
            self.transform = T.ToTensor()

        with Image.open(files[0]) as img:
            self.raw_img_size = img.size
            self.num_ch = get_num_channels(mode=img.mode)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img)


def get_digits_ds():
    digits = Digits()
    return DataLoader(dataset=digits, batch_size=5, shuffle=True)


def get_celeb_ds(folders, max_mb=100, batch_size=5) -> DataLoader:
    files = get_files(directories=folders, max_mb=max_mb)    
    ds = ImageDataset(files=files)
    return DataLoader(dataset=ds, batch_size=batch_size, shuffle=True)

    
def get_datasets(res: ImgRunTracker) -> DataLoader:
    files = get_files(**res.cfg['data'])
    train_n = min(len(files), int(round(len(files) * res.cfg['dataset']['train_ratio'])))
    train_files = files[0:train_n]
    val_files = files[train_n:]
    crop_size = res.cfg['dataset']['crop_size']
    return dict(train=CroppedImageDataset(files=train_files, crop_size=crop_size),
                val=CroppedImageDataset(files=val_files, crop_size=crop_size))


def resize_and_save(input_dir: str, output_dir: str, num_files=10000, res: Tuple[int, int] = (64, 64), train_perc=0.8) -> None:
    import shutil
    from tqdm import tqdm

    files = get_files(directories=[input_dir], max_mb=9e99)
    random.shuffle(files)
    files = files[0:num_files]
    n_train = int(round(train_perc * len(files)))

    splits = dict(train=files[0:n_train], val=files[n_train:])
    for split, s_files in splits.items():
        s_dir = os.path.join(output_dir, split)
        if os.path.exists(s_dir):
            print(f"Removing existing dir: {s_dir}")
            shutil.rmtree(s_dir)
        os.makedirs(s_dir, exist_ok=True)
        for fname in tqdm(s_files, desc=f"Processing split {split}", ncols=100):
            img = Image.open(fname).convert("RGB")
            img = img.resize(size=res, resample=Image.BICUBIC)
            o_pth = os.path.join(output_dir, split, os.path.basename(fname))
            img.save(o_pth)
