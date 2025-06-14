import os
from pprint import pprint
from pathlib import Path
import yaml
import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from imagegen.utils import get_timestamp


def load_config(path) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    root_path = find_root()
    return replace_root_in_dict(d=cfg, root_path=root_path)


def find_root(project_name="imagegen") -> str:
    # Get absolute path to current file
    current_file = Path(__file__).resolve()

    # Make sure we are where we think we are
    assert current_file.name == "setup.py", f"Unexpected file: {current_file.name}"
    assert (
        current_file.parent.name == project_name
    ), f"Expected '{project_name}', got {current_file.parent.name}"
    assert (
        current_file.parent.parent.name == "src"
    ), f"Expected 'src', got {current_file.parent.parent.name}"

    return current_file.parent.parent.parent


def replace_root_in_dict(d: dict, root_path: str) -> dict:
    """Recursively replace '{{root}}' with root_path in all string values."""
    if isinstance(d, dict):
        return {k: replace_root_in_dict(v, root_path) for k, v in d.items()}
    elif isinstance(d, list):
        return [replace_root_in_dict(item, root_path) for item in d]
    elif isinstance(d, str):
        return d.replace("{{root}}", str(root_path))
    else:
        return d


def safe_open(file_path, mode="w"):
    """Ensure parent directory exists, then open the file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, mode)


class DenoisingDiffusionRunTracker:
    __slots__ = [
        "cfg",
        "train_ds",
        "val_ds",
        "train_dl",
        "val_dl",
        "unet",
        "optimizer",
        "trainer",
        "timestamp",
        "logdir",
        "savedir",
        "train_writer",
        "val_writer",
    ]

    def __init__(self, cfg_fname: str):
        cfg = load_config(path=cfg_fname)
        self.cfg = cfg
        self.train_ds = None
        self.val_ds = None
        self.train_dl = None
        self.val_dl = None
        self.unet = None
        self.optimizer = None
        self.trainer = None
        self.timestamp = None
        self.logdir = None
        self.savedir = None
        self.train_writer = None
        self.val_writer = None

        if self.cfg["verbose"]:
            print("--- CONFIG --")
            pprint(self.cfg)


def save_model_and_meta(
    cfg: dict, savedir: str, model: nn.Module, optimizer, epoch: int = 0
) -> str:
    if not cfg["train"]["save"]:
        return ""

    assert savedir, "save dir not set"

    cfg_fname = os.path.join(savedir, "config.yaml")
    with open(cfg_fname, "w") as f:
        yaml.dump(cfg, f, sort_keys=False, default_flow_style=False, indent=2)

    checkpoint_pth = os.path.join(savedir, f"model_epoch{epoch:05d}.pt")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_pth,
    )
    print(f"Saved model to {checkpoint_pth}")


def setup_training(cfg: dict) -> dict:
    # Get timestamp and create log directory
    timestamp = get_timestamp()
    res = dict(
        logdir=cfg["logdir"].format(timestamp=timestamp),
        savedir=cfg["savedir"].format(timestamp=timestamp),
    )
    os.makedirs(res["logdir"], exist_ok=True)
    os.makedirs(res["savedir"], exist_ok=True)
    res.update(
        dict(
            train_writer=SummaryWriter(res["logdir"] + "/train"),
            val_writer=SummaryWriter(res["logdir"] + "/val"),
        )
    )
    return res


def setup_optim(model: nn.Module, cfg: dict) -> optim.Optimizer:
    name = cfg["name"].lower()
    optimizer = None
    args = dict(lr=cfg["learning_rate"])
    if name == "adam":
        optimizer = optim.Adam
        args.update(cfg["adam_args"])
    elif name == "adamw":
        optimizer = optim.AdamW
        args.update(cfg["adam_args"])

    if optimizer is None:
        raise ValueError(f"unknown optimizer name: {name}")

    return optimizer(model.parameters(), **args)
