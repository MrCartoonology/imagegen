import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.utils as nn_utils

# from tqdm.notebook import tqdm
from tqdm import tqdm
from imagegen.setup import setup_training, setup_optim, save_model_and_meta
import imagegen.utils as ig_utils


class DDPMTrainer(nn.Module):
    def __init__(self, cfg: dict):
        super(DDPMTrainer, self).__init__()
        self.cfg = cfg
        noise_schedule_cfg = cfg["diffusion_noise_schedule"]
        beta_start = noise_schedule_cfg["beta_start"]
        beta_end = noise_schedule_cfg["beta_end"]
        self.TT = noise_schedule_cfg["num_timesteps"]
        betas = torch.linspace(beta_start, beta_end, self.TT)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def train(self, train_dl: DataLoader, val_dl: DataLoader, unet: nn.Module):

        if self.cfg["verbose"]:
            print("------- Training -------")
        tr_cfg = self.cfg["train"]
        total_steps = tr_cfg["epochs"] * len(train_dl)
        progress = tqdm(total=total_steps, desc="Training", ncols=100)

        unet = unet.to(self.cfg["device"])
        self.to(self.cfg["device"])

        optimizer = setup_optim(model=unet, cfg=self.cfg["optim"])
        optimizer.zero_grad()

        train_res = setup_training(cfg=tr_cfg)
        train_writer, val_writer = train_res["train_writer"], train_res["val_writer"]

        step = 0
        for epoch in range(tr_cfg["epochs"]):
            for x in train_dl:
                x = x.to(self.cfg["device"])
                train_loss = self.train_step(x0=x, mdl=unet, optimizer=optimizer)
                train_writer.add_scalar("loss", train_loss.item(), step)

                progress.set_description(
                    f"Training epoch={epoch:3d} step={step:8d} loss={train_loss.item():8.5f}"
                )
                progress.update(1)
                step += 1
                if tr_cfg["max_steps"] > 0 and step >= tr_cfg["max_steps"]:
                    print("max steps set - stopping early")
                    return train_res
        return train_res

    def train_step(self, x0, mdl, optimizer):
        batch_size = x0.size(0)
        self.cfg["train"]
        t = torch.randint(0, self.TT, (batch_size,), device=x0.device)
        eps = torch.randn_like(x0, device=x0.device)
        alpha_bars_t = self.alpha_bars[t]
        x0_coeff = torch.sqrt(alpha_bars_t).view(batch_size, 1, 1, 1)
        eps_coeff = torch.sqrt(1.0 - alpha_bars_t).view(batch_size, 1, 1, 1)
        pred_eps = mdl(x0_coeff * x0 + eps_coeff * eps, t)
        loss = nn.functional.mse_loss(pred_eps, eps)
        loss.backward()
        nn_utils.clip_grad_norm_(mdl.parameters(), self.cfg["train"]["max_grad_norm"])
        optimizer.step()
        optimizer.zero_grad()
        return loss
