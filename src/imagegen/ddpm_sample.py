from tqdm import tqdm
import numpy as np
import torch
from matplotlib import animation, pyplot as plt

from imagegen.setup import find_root, load_config
import imagegen.unet as unet
import imagegen.train as train
import imagegen.data as data
import imagegen.eval as eval


CFG_FNAME = find_root() / "config/ddpm_config.yaml"


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
        for i in tqdm(range(B), desc="Rendering histograms", ncols=100):
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
        for i in tqdm(range(B), desc="Rendering images", ncols=100):
            yield i

    def update(i):
        ax.clear()
        timestep = B - 1 - i
        ax.imshow(
            basic_decode(arr[timestep]),
            aspect='auto',
            interpolation='nearest',
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


def make_samples(mdl, epoch, cfg: dict, clip_noise: bool = True):
    ddpm = train.DDPMTrainer(cfg=cfg)
    x0_T = ddpm.sample(mdl=mdl, clip_noise=clip_noise)
#    decode_and_show_percentile(x0_T[0], title=f"Sample at T=0, epoch {epoch}")
#    make_histogram_movie(x0_T, out_path=f"sample_hist_epoch{epoch:05d}.mp4")
    make_image_movie(x0_T, out_path=f"sample_epoch{epoch:05d}.mp4")
    return x0_T


def load_model(cfg: dict, saved_model: str):
    mdl = unet.UNet()
    state_dict = torch.load(saved_model, map_location=cfg["device"])
    mdl.load_state_dict(state_dict["model_state_dict"])
    mdl.eval()
    return mdl.to(cfg["device"])


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

