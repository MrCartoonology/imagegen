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


def create_autoencoder(res: ImgRunTracker) -> nn.Module:
    cfg = res.cfg
    device = get_device(device=cfg["device"])
    ae = None
    print(model)
    total, trainable, size_mb = count_parameters(model)
    print(f"  Total params:     {total:,}")
    print(f"  Trainable params: {trainable:,}")
    print(f"  Model size:       {size_mb:.2f} MB")
    return model


class AutoEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
