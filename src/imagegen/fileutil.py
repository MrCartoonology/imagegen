from pathlib import Path
import random
from time import sleep
from typing import Iterable, Iterator, Tuple
from PIL import Image
from IPython.display import HTML
from IPython.display import display, clear_output
import matplotlib.pyplot as plt


# ────────────────────────────────────────────────────────────────────────────────
# 1. Core generator ─ yields (Path, PIL.Image) pairs
# ────────────────────────────────────────────────────────────────────────────────
def iter_images(
    folder: str | Path,
    patterns: Iterable[str] = ("*.jpg", "*.jpeg", "*.png", "*.gif"),
    recursive: bool = False,
    shuffle: bool = False,
) -> Iterator[Tuple[Path, Image.Image]]:
    """
    Lazily walk *folder* and yield `(path, image)` for every matching file.

    The image is `.copy()`‑ed so it stays valid after the context manager exits.
    """
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} is not a directory")

    paths = []
    for pat in patterns:
        globber = folder.rglob if recursive else folder.glob
        paths.extend(list(globber(pat)))
    
    if shuffle:
        random.shuffle(paths)    
    
    for img_path in paths:
        try:
            with Image.open(img_path) as im:
                yield img_path, im.copy()
        except Exception as e:
            print(f"⚠️  Skipping {img_path.name}: {e}")


def scale_image(filename: str, resolutions=(64, 32)):
    # Load image
    img = Image.open(filename)

    # Save full-resolution copy
    original_path = Path("img_fullres.png")
    img.save(original_path)

    # HTML sections
    html_blocks = [f"""
        <div>
            <p>Full Resolution</p>
            <img src="{original_path}" style="image-rendering: pixelated;">
        </div>
    """]

    # Downsample and save at each resolution
    for res in resolutions:
        img_small = img.resize((res, res), Image.BICUBIC)
        out_path = Path(f"img_{res}.png")
        img_small.save(out_path)
        html_blocks.append(f"""
            <div>
                <p>Downsampled ({res}×{res})</p>
                <img src="{out_path}" style="image-rendering: pixelated; width:{res}px; height:{res}px;">
            </div>
        """)

    # Display all blocks
    display(HTML(f"""
        <div style="display: flex; gap: 20px;">
            {''.join(html_blocks)}
        </div>
    """))


def display_image_grid(folder: str | Path, nr: int = 2, nc: int = 6, width=2, shuffle=True):
    images = []
    for _, img in iter_images(folder, shuffle=shuffle):
        images.append(img)
        if len(images) > nr * nc:
            break

    fig, axs = plt.subplots(nr, nc, figsize=(nc * width, nr * width), squeeze=False)
    for row in axs:
        for ax in row:
            ax.imshow(images.pop(0))
            ax.axis("off")

    plt.tight_layout()
    plt.show()


# ────────────────────────────────────────────────────────────────────────────────
# 2. Slideshow helper ─ displays each image then waits *delay* second#
def preview_slideshow(
    folder: str | Path,
    delay: float = 2.0,
    lim: int = 0,
    **iter_kw,   # forwarded to iter_images (patterns=…, recursive=…)
):
    """
    Show every image in *folder* inside a Jupyter notebook with a timed pause.
    """
    count = 0
    for path, img in iter_images(folder, **iter_kw):
        clear_output(wait=True)
        print(f"{path.name}: {img.width} × {img.height} px")
        display(img)
        sleep(delay)
        count += 1
        if lim > 0 and count > lim:
            break

    clear_output(wait=True)
    print(f"Done! Displayed {count} images.")


# ────────────────────────────────────────────────────────────────────────────────
# 3. Minimum‑size scanner ─ returns (min_width, min_height)
# ────────────────────────────────────────────────────────────────────────────────
def min_image_size(
    folder: str | Path,
    **iter_kw,
) -> Tuple[int, int]:
    """
    Sweep through images in *folder* and return the smallest width & height found.
    """
    min_w = float("inf")
    min_h = float("inf")
    found = 0

    for _, img in iter_images(folder, **iter_kw):
        min_w = min(min_w, img.width)
        min_h = min(min_h, img.height)
        found += 1

    if found == 0:
        raise ValueError("No matching images found.")
    return int(min_w), int(min_h)
