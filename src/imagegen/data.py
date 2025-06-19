import os
import random
import shutil
from tqdm import tqdm
from typing import List, Tuple

from PIL import Image

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader


def get_files(directories: List[str]) -> List[str]:
    all_files = []
    for directory in directories:
        for root, _, files in os.walk(directory):
            all_files.extend([os.path.join(root, f) for f in files])
    random.shuffle(all_files)
    return all_files


def preprocess(cfg: dict, verbose=False):
    if verbose:
        print("-------- PREPROCESS/RESIZE IMAGES ----------")
    input_dir = cfg["raw_data_dir"]
    output_dir = cfg["resized_dir"]
    if cfg["reprocess"] or not os.path.exists(output_dir):
        resize_and_save(
            input_dir=input_dir,
            output_dir=output_dir,
            num_files=cfg["num_files"],
            res=tuple(cfg["resize"]),
            train_perc=cfg["train_perc"],
        )
    else:
        if verbose:
            print("preprocessing already done.")


def resize_and_save(
    input_dir: str,
    output_dir: str,
    num_files=10000,
    res: Tuple[int, int] = (64, 64),
    train_perc=0.8,
) -> None:
    files = get_files(directories=[input_dir])
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


class CachedImageDataset(Dataset):
    def __init__(self, dir: str, limit_files: int = 0, verbose: bool = False):
        self.files = get_files(directories=[dir])
        if limit_files:
            random.shuffle(self.files)
            self.files = self.files[0:limit_files]
            if verbose:
                print(f"Limited files to {limit_files} from {dir}")
        self.transform = T.Compose(
            [
                T.ToTensor(),  # (0,1)
                T.Lambda(lambda x: (x - 0.5) * 2),
            ]
        )

        # Read one image to get shape
        sample_img = Image.open(self.files[0]).convert("RGB")
        sample_tensor = self.transform(sample_img)
        channels, height, width = sample_tensor.shape

        # Preallocate storage
        self.data = torch.empty(
            (len(self.files), channels, height, width), dtype=sample_tensor.dtype
        )

        # Fill in the preallocated tensor
        for i, path in enumerate(tqdm(self.files, desc="Caching images", ncols=100)):
            img = Image.open(path).convert("RGB")
            self.data[i] = self.transform(img)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.data[idx]


def get_cached_image_dataset(split: str, verbose: bool, root_dir: str) -> Dataset:
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    if verbose:
        print(f"------ CACHING DATASET {split} -------")
    return CachedImageDataset(dir=root_dir + f"/{split}")


def get_cached_image_datasets(verbose: bool, root_dir: str) -> Tuple[Dataset, Dataset]:
    train_ds = get_cached_image_dataset(
        split="train", verbose=verbose, root_dir=root_dir
    )
    val_ds = get_cached_image_dataset(split="val", verbose=verbose, root_dir=root_dir)
    return train_ds, val_ds


def get_dataloader(
    split: str, verbose: bool, batch_size: int, ds: Dataset, shuffle: bool = True
) -> DataLoader:
    if verbose:
        print(f" -- Creating DataLoader {split} --")
    return DataLoader(dataset=ds, batch_size=batch_size, shuffle=shuffle)


def get_dataloaders(
    verbose: bool, batch_size: int, train_ds: Dataset, val_ds: Dataset
) -> Tuple[DataLoader, DataLoader]:
    train_dl = get_dataloader(
        split="train", verbose=verbose, batch_size=batch_size, ds=train_ds, shuffle=True
    )

    val_dl = get_dataloader(
        split="val", verbose=verbose, batch_size=batch_size, ds=val_ds, shuffle=True
    )
    return train_dl, val_dl
