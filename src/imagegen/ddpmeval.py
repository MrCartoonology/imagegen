from typing import Union
import numpy as np
import torch
import torch.nn as nn
from matplotlib import animation, pyplot as plt
from collections import defaultdict

from imagegen.setup import find_root, load_config
import imagegen.unet as unet
import imagegen.train as train
import imagegen.data as data


from torch.utils.data import DataLoader
from imagegen.train import DDPMTrainer
from imagegen.unet import UNet
from imagegen.ddpmhuggingface import Unet as UNetHF


def get_tqdm():
    from IPython import get_ipython
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
    except NameError:
        from tqdm import tqdm
    return tqdm


CFG_FNAME = find_root() / "config/ddpm_config.yaml"


def image_channel_histograms(dl, clip=False):
    arr = np.concatenate([x0.detach().cpu().numpy() for x0 in dl])
    plt.figure(figsize=(12, 4))
    for i, ch in zip([0, 1, 2], ["R", "G", "B"]):
        plt.subplot(1, 3, i + 1)
        charr = arr[:, i, :, :]
        if clip:
            charr = np.clip(charr, -1.0, 1.0)
        plt.hist(charr.flatten(), bins=300)
        plt.title(f"ch {ch}")
    plt.show()


def predict_bias(ddpm_train: DDPMTrainer, mdl: UNet, val_dl: DataLoader):
    mdl.eval()
    with torch.no_grad():
        mu = torch.zeros(size=(1 + ddpm_train.TT,))
        for x0 in tdqm(val_dl, desc="Evaluating bias", ncols=100):
            for t in range(ddpm_train.TT, 0, -1):
                tTT = torch.full((x0.size(0),), t, device=x0.device)
                pred_eps, eps, _, xt = ddpm_train.predict_noise(x0=x0, mdl=mdl, t=tTT)
                import IPython

                IPython.embed()
                mu[t] += pred_eps.mean()
        mu /= len(val_dl.dataset)
        return mu


def make_histogram_movie(x0T, out_path="hist_movie.mp4", bins=100):
    if hasattr(x0T, "detach"):
        arr = x0T.detach().cpu().numpy()
    else:
        arr = np.array(x0T)

    B = arr.shape[0]
    flat_all = arr.reshape(B, -1)

    global_min = flat_all.min()
    global_max = flat_all.max()

    fig, ax = plt.subplots(figsize=(8, 6))

    # Define a generator that yields frames and updates tqdm
    def frame_generator():
        for i in get_tqdm()(range(B), desc="Rendering histograms", ncols=100):
            yield i

    def update(i):
        ax.clear()
        timestep = B - 1 - i
        ax.hist(
            flat_all[timestep],
            bins=bins,
            range=(global_min, global_max),
            color="skyblue",
            edgecolor="black",
        )
        ax.set_title(f"Histogram at t={timestep}", fontsize=14)
        ax.set_xlim(global_min, global_max)
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")

    ani = animation.FuncAnimation(fig, update, frames=frame_generator, repeat=False)
    ani.save(out_path, writer="ffmpeg", fps=100)
    plt.close(fig)


def make_image_movie(x0T, out_path):
    if hasattr(x0T, "detach"):
        arr = x0T.detach().cpu().numpy()
    else:
        arr = np.array(x0T)

    B = arr.shape[0]
    fig, ax = plt.subplots(figsize=(8, 6))

    # Define a generator that yields frames and updates tqdm
    def frame_generator():
        for i in get_tqdm()(range(B), desc="Rendering images", ncols=100):
            yield i

    def update(i):
        ax.clear()
        timestep = B - 1 - i
        ax.imshow(
            basic_decode(arr[timestep]),
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_title(f"Image at t={timestep}", fontsize=14)

    ani = animation.FuncAnimation(fig, update, frames=frame_generator, repeat=False)
    ani.save(out_path, writer="ffmpeg", fps=100)
    plt.close(fig)


def basic_decode(x):
    """
    Decode a (3, H, W) tensor to a uint8 image using basic clipping.
    Args:
        x (torch.Tensor): shape (3, H, W), likely unbounded
    Returns:
        np.ndarray: decoded image in shape (H, W, 3)
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    assert x.shape[0] == 3, "Expected shape (3, H, W)"

    img = np.clip(x.transpose(1, 2, 0), -1.0, 1.0)
    img = ((img + 1.0) * 127.5).astype(np.uint8)

    return img


def decode_and_show_percentile(x, title=None):
    """
    Decode a (3, H, W) tensor to a uint8 image using per-channel 1st–99th percentile scaling.
    Args:
        x (torch.Tensor): shape (3, H, W), likely unbounded
        title (str): optional title
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    assert x.shape[0] == 3, "Expected shape (3, H, W)"

    img = np.zeros_like(x)

    for c in range(3):
        channel = x[c]
        aa = np.percentile(channel, 1)
        bb = np.max(channel)

        channel2 = np.clip(channel, aa, bb) - aa
        channel2 /= bb - aa
        channel2 *= 256
        img[c] = np.clip(channel2, 0.0, 255.0)

    img = img.transpose(1, 2, 0).astype(np.uint8)

    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()


def make_samples(
    mdl,
    epoch,
    cfg: dict,
    clip_noise: bool = False,
    clip_pred: bool = False,
    clip_val: float = 1.15,
):
    ddpm = train.DDPMTrainer(cfg=cfg)
    x0_T = ddpm.sample(
        mdl=mdl, clip_noise=clip_noise, clip_pred=clip_pred, clip_val=clip_val
    )
    #    decode_and_show_percentile(x0_T[0], title=f"Sample at T=0, epoch {epoch}")
    #    make_histogram_movie(x0_T, out_path=f"sample_hist_epoch{epoch:05d}.mp4")
    make_image_movie(x0_T, out_path=f"sample_epoch{epoch:05d}.mp4")
    return x0_T


def load_model(cfg: dict, saved_model: str):
    if cfg['unet'] == 'huggingface':
        mdl = UNetHF(dim=64)
    else:
        mdl = unet.UNet()
    state_dict = torch.load(saved_model, map_location=cfg["device"])
    mdl.load_state_dict(state_dict["model_state_dict"])
    mdl.eval()
    return mdl.to(cfg["device"])


def eval_stats_per_t(
    mdl: unet.UNet,
    dl: DataLoader,
    cfg: dict,
    t_lim: Union[None, tuple] = None,
    num_batches: int = 0,
) -> dict:
    ddpm = train.DDPMTrainer(cfg=cfg).to(cfg["device"])
    mdl.eval()

    if t_lim is None:
        t_range = list(range(ddpm.TT, 0, -1))
    else:
        assert len(t_lim) == 2, "t_lim must be a tuple of length 2"
        tb = max(t_lim)
        ta = min(t_lim)
        t_range = list(range(tb, ta - 1, -1))

    loss_t = defaultdict(list)
    pred_eps_t_mu = defaultdict(list)
    pred_eps_t_std = defaultdict(list)

    if num_batches <= 0:
        num_batches = len(dl)

    total_steps = len(t_range) * num_batches
    pbar = get_tqdm()(total=total_steps, desc="Evaluating time based model stats: ")

    with torch.no_grad():
        for bi, x0 in enumerate(dl):
            if bi >= num_batches:
                break
            x0 = x0.to(cfg["device"])
            for t in t_range:
                t_tensor = torch.full((x0.size(0),), t, device=x0.device)
                pred_eps, eps, _, xt = ddpm.predict_noise(x0=x0, mdl=mdl, t=t_tensor)
                loss = nn.functional.mse_loss(pred_eps, eps, reduction='none')
                loss = loss.view(loss.size(0), -1).mean(dim=1)  # shape: [B]
                loss_t[t].append(loss.detach().cpu().numpy())
                pred_eps_t_mu[t].append(pred_eps.view(pred_eps.size(0), -1).mean(dim=1).detach().cpu().numpy())
                pred_eps_t_std[t].append(pred_eps.view(pred_eps.size(0), -1).std(dim=1).detach().cpu().numpy())
                pbar.update()
    pbar.close()
    for t in loss_t.keys():
        loss_t[t] = np.stack(loss_t[t], axis=0)
        pred_eps_t_mu[t] = np.stack(pred_eps_t_mu[t], axis=0)
        pred_eps_t_std[t] = np.stack(pred_eps_t_std[t], axis=0)

    return dict(
        loss_t=loss_t, pred_eps_t_mu=pred_eps_t_mu, pred_eps_t_std=pred_eps_t_std
    )


def plot_stats_per_t_loss(
        t_stats: dict, figsize=(10,4), val_loss=0.0943
):
    ts = sorted(t_stats["loss_t"].keys())
    y = [t_stats['loss_t'][t].mean() for t in ts]
    plt.figure(figsize=figsize)
    plt.plot(ts, y, label='loss per timestep')
    if val_loss:
        plt.plot([ts[0], ts[-1]], [val_loss, val_loss], color='r', linestyle='--', label='loss')
    plt.title("Loss per t")
    plt.xlabel("t - timestep")
    plt.ylabel("mse loss")
    plt.legend()
    plt.show()


def plot_mu_std_per_t(t_stats: dict, figsize=(10, 4), label="pred_eps mean ± std"):
    ts = sorted(t_stats["pred_eps_t_mu"].keys())
    mu = np.array([t_stats["pred_eps_t_mu"][t].mean() for t in ts])
    std = np.array([t_stats["pred_eps_t_std"][t].mean() for t in ts])

    plt.figure(figsize=figsize)
    plt.plot(ts, mu, label=label)
    plt.fill_between(ts, mu - std, mu + std, alpha=0.3)
    plt.plot([ts[0], ts[-1]], [0, 0], color='r', linestyle='--', label='zero - unbiased')
    plt.plot([ts[0], ts[-1]], [1, 1], color='g', linestyle='--', label='std +- 1')
    plt.plot([ts[0], ts[-1]], [-1, -1], color='g', linestyle='--')
    plt.title("Mean of predicted ε per timestep with std")
    plt.xlabel("t - timestep")
    plt.ylabel("mean(pred_eps)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    import sys

    print(sys.argv)
    assert len(sys.argv) == 2
    epoch = int(sys.argv[1])
    saved_model = f"saved_models/20250612-2155/model_epoch{epoch:05d}.pt"
    cfg = load_config(CFG_FNAME)
    # cfg["device"] = "cpu"
    mdl = load_model(cfg=cfg, saved_model=saved_model)
    x0T = make_samples(cfg=cfg, mdl=mdl, epoch=epoch, clip_noise=True)
    #    make_image_movie(x0T, out_path=f"images_epoch{epoch:05d}.mp4")
    sys.exit(0)
    val_ds = data.get_cached_image_dataset(
        split="val",
        verbose=True,
        root_dir=cfg["data"]["resized_dir"],
    )
    val_dl = data.get_dataloader(
        split="val",
        verbose=cfg["verbose"],
        batch_size=cfg["data"]["batch_size"],
        ds=val_ds,
    )
