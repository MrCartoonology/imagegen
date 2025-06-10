import torch
import torch.nn as nn
from imagegen.setup import ImgRunTracker
from imagegen.utils import get_device


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_mb = total_bytes / (1024**2)
    return total_params, trainable_params, total_mb


class DDPMUnet(nn.Module):
    """unet for ddpm. """
    def __init__(self, cfg: dict):
        super().__init__()

    def forward(self, x):
        return x
