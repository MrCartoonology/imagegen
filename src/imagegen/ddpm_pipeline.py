from imagegen.setup import find_root, DenoisingDiffusionRunTracker, setup_training
import imagegen.data as data
import imagegen.unet as unet
import imagegen.train as train


CFG_FNAME = find_root() / "config/ddpm_config.yaml"


def run(cfg_fname: str = CFG_FNAME) -> DenoisingDiffusionRunTracker:
    res = DenoisingDiffusionRunTracker(cfg_fname=cfg_fname)
    data.preprocess(cfg=res.cfg["data"], verbose=res.cfg["verbose"])
    res.train_ds, res.val_ds = data.get_cached_image_datasets(
        verbose=res.cfg["verbose"],
        root_dir=res.cfg["data"]["resized_dir"],
    )
    res.train_dl, res.val_dl = data.get_dataloaders(
        verbose=res.cfg["verbose"],
        batch_size=res.cfg["data"]["batch_size"],
        train_ds=res.train_ds,
        val_ds=res.val_ds,
    )
    if res.cfg["verbose"]:
        print("------- Creating UNET -------")
    res.unet = unet.UNet()
    res.trainer = train.DDPMTrainer(cfg=res.cfg)
    training_res = res.trainer.train(
        train_dl=res.train_dl,
        val_dl=res.val_dl,
        unet=res.unet,
    )
    for k, v in training_res.items():
        setattr(res, k, v)


if __name__ == "__main__":
    run()
