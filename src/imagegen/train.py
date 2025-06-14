import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.utils as nn_utils

# from tqdm.notebook import tqdm
from tqdm import tqdm, trange
from imagegen.setup import setup_training, setup_optim, save_model_and_meta
import imagegen.utils as ig_utils


class DDPMTrainer(nn.Module):
    def __init__(self, cfg: dict):
        super(DDPMTrainer, self).__init__()
        self.cfg = cfg
        self.img_H, self.img_W = cfg['data']['resize']

        noise_schedule_cfg = cfg["diffusion_noise_schedule"]
        beta_start = noise_schedule_cfg["beta_start"]
        beta_end = noise_schedule_cfg["beta_end"]
        self.TT = noise_schedule_cfg["num_timesteps"]

        betas = torch.linspace(beta_start, beta_end, self.TT)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        # we want to index as in the paper - 1-up, not 0-up
        betas = torch.cat([betas[:1], betas])
        alphas = torch.cat([alphas[:1], alphas])
        alpha_bars = torch.cat([alpha_bars[:1], alpha_bars])
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

        epoch_save_interval = int(round(tr_cfg['epochs'] / tr_cfg['num_models_save'])) if tr_cfg['save'] else 0
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
            val_loss = self.evaluate(mdl=unet, val_dl=val_dl)
            val_writer.add_scalar("loss", val_loss.item(), step)
            if (epoch > 0) and (epoch < tr_cfg["epochs"] - 1) and epoch_save_interval and (epoch % epoch_save_interval == 0):
                save_model_and_meta(cfg=self.cfg, savedir=train_res["savedir"], model=unet, optimizer=optimizer, epoch=epoch)
#            self.sample(mdl=unet)
        
        if tr_cfg['save']:
            save_model_and_meta(cfg=self.cfg, savedir=train_res["savedir"], model=unet, optimizer=optimizer, epoch=epoch)

        return train_res

    def evaluate(self, mdl, val_dl):
        mdl.eval()
        n = 0
        loss = 0.0
        with torch.no_grad():
            for x0 in val_dl:
                x0 = x0.to(self.cfg["device"])
                pred_eps, eps, _, _ = self.predict_noise(x0=x0, mdl=mdl)
                loss += nn.functional.mse_loss(pred_eps, eps)
                n += 1
        loss /= n
        mdl.train()
        return loss

    def predict_noise(self, x0, mdl):
        batch_size = x0.size(0)
        t = torch.randint(1, 1 + self.TT, (batch_size,), device=x0.device)
        eps = torch.randn_like(x0, device=x0.device)
        alpha_bars_t = self.alpha_bars[t]
        x0_coeff = torch.sqrt(alpha_bars_t).view(batch_size, 1, 1, 1)
        eps_coeff = torch.sqrt(1.0 - alpha_bars_t).view(batch_size, 1, 1, 1)
        mdl_input = x0_coeff * x0 + eps_coeff * eps
        pred_eps = mdl(mdl_input, t)
        return pred_eps, eps, t, mdl_input

    def train_step(self, x0, mdl, optimizer):
        mdl.train()
        pred_eps, eps, _, _ = self.predict_noise(x0=x0, mdl=mdl)
        loss = nn.functional.mse_loss(pred_eps, eps)
        loss.backward()
        nn_utils.clip_grad_norm_(mdl.parameters(), self.cfg["train"]["max_grad_norm"])
        optimizer.step()
        optimizer.zero_grad()
        return loss
    
    def sample(self, mdl):
        z0_T = torch.randn(size=(1 + self.TT, 3, self.img_H, self.img_W)).to(self.cfg['device'])
        z0_T[0] = 0
        z0_T[1] = 0
        x0_T = torch.empty_like(z0_T).to(self.cfg['device'])
        x0_T[self.TT] = z0_T[self.TT]

        pred_coeff = (1.0 - self.alphas) / torch.sqrt(1.0 - self.alpha_bars)
        pair_coeff = 1.0 / torch.sqrt(self.alphas)
        sigma = torch.sqrt(self.betas)

        for t in trange(self.TT, 0, -1, desc="Sampling", ncols=100):
            z = z0_T[t - 1]
            xt = x0_T[t]
            t_tensor = torch.tensor(t).view(size=(1,)).to(self.cfg['device'])
            eps_pred = mdl(x=xt, t=t_tensor)[0]
            pair = (xt - pred_coeff[t] * eps_pred)
            x0_T[t - 1] = pair_coeff[t] * pair + sigma[t] * z

        return x0_T

