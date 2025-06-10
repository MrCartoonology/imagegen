import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.utils as nn_utils
from imagegen.setup import (
    RunTracker,
    setup_training,
    setup_optim,
    save_model_and_meta
)
import imagegen.utils as ig_utils


def train_ddpm(cfg: dict, mdl: nn.Module, train_dl: DataLoader) -> RunTracker:
#    res = setup_training(res)

#    model = res.model
#    train_cfg = res.cfg["train"]
#    eval_cfg = res.cfg["eval"]
#    prompt_cfg = res.cfg["prompt"]

    device = ig_utils.get_device(cfg["device"])

    T = cfg['num_diffusion_timesteps']
    img_size = train_dl.dataset.raw_img_size
    num_ch = train_dl.dataset.num_ch
    print(img_size, num_ch)
    for batch in train_dl:
        batch = batch.to(device)
        batch_size = batch.size(0)  # or len(batch) if batch is a list
        t = torch.randint(0, T, (batch_size,), device=batch.device)
        eps = torch.randn_like(batch)

        y = mdl(batch)
        print(img_size, num_ch, batch.size())
        return
