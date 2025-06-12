# from  https://github.com/jmtomczak/intro_dgm

import numpy as np
import torch
import torch.functional as F


PI = torch.from_numpy(np.asarray(np.pi))
EPS = 1.0e-7


def log_categorical(x, p, num_classes=256, reduction=None, dim=None):
    x = x.long()
    assert len(x.size()) == len(p.size())
    log_p = torch.log(torch.clamp(p, EPS, 1.0 - EPS))
    selected_log_p = torch.gather(input=log_p, dim=1, index=x)
    if reduction == "avg":
        return torch.mean(selected_log_p, dim)
    elif reduction == "sum":
        return torch.sum(selected_log_p, dim)
    else:
        return selected_log_p


def log_bernoulli(x, p, reduction=None, dim=None):
    pp = torch.clamp(p, EPS, 1.0 - EPS)
    log_p = x * torch.log(pp) + (1.0 - x) * torch.log(1.0 - pp)
    if reduction == "avg":
        return torch.mean(log_p, dim)
    elif reduction == "sum":
        return torch.sum(log_p, dim)
    else:
        return log_p


def log_normal_diag(x, mu, log_var, reduction=None, dim=None):
    D = x.shape[1]
    log_p = (
        -0.5 * D * torch.log(2.0 * PI)
        - 0.5 * log_var
        - 0.5 * torch.exp(-log_var) * (x - mu) ** 2.0
    )
    if reduction == "avg":
        return torch.mean(log_p, dim)
    elif reduction == "sum":
        return torch.sum(log_p, dim)
    else:
        return log_p


def log_standard_normal(x, reduction=None, dim=None):
    D = x.shape[1]
    log_p = -0.5 * D * torch.log(2.0 * PI) - 0.5 * x**2.0
    if reduction == "avg":
        return torch.mean(log_p, dim)
    elif reduction == "sum":
        return torch.sum(log_p, dim)
    else:
        return log_p
