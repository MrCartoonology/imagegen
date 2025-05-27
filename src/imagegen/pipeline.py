import glob
import imagegen.data as data
from imagegen.setup import find_root, ImgRunTracker

CFG_FNAME = find_root() / "config/config.yaml"


def run(cfg_fname: str = CFG_FNAME) -> ImgRunTracker:
    res = ImgRunTracker(cfg_fname=cfg_fname)
    res.datasets = data.get_datasets(res=res)
    res.model = 
    return res


if __name__ == '__main__':
    run()