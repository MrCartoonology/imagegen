import datetime
import torch
import random
import numpy as np


def get_timestamp():
    """Get current timestamp in YYYYMMDD-HHMM format"""
    return datetime.datetime.now().strftime("%Y%m%d-%H%M")


def get_device(device="auto"):
    """
    Get the best available device for PyTorch operations.

    Args:
        device (str): Desired device. If 'auto', will automatically select the best available device.
                     Options: 'auto', 'cpu', 'cuda', 'mps'

    Returns:
        str: The device to use ('cpu', 'cuda', or 'mps')
    """
    if device == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    return device


def seed_everything(seed: int, device: str) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

