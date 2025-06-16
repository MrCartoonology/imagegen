from tqdm import tqdm as tdqm
import torch
from torch.utils.data import DataLoader
from imagegen.train import DDPMTrainer
from imagegen.unet import UNet


def predict_bias(ddpm_train: DDPMTrainer, mdl: UNet, val_dl: DataLoader):
    mdl.eval()
    with torch.no_grad():
        mu = torch.zeros(size=(1+ddpm_train.TT,))
        for x0 in tdqm(val_dl, desc="Evaluating bias", ncols=100):
            for t in range(ddpm_train.TT, 0, -1):
                tTT = torch.full((x0.size(0),), t, device=x0.device)
                pred_eps, eps, _, xt = ddpm_train.predict_noise(x0=x0, mdl=mdl, t=tTT)
                import IPython
                IPython.embed()
                mu[t] += pred_eps.mean()
        mu /= len(val_dl.dataset)
        return mu

