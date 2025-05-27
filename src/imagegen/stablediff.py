

from datasets import load_dataset
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt


def load_pokemon_dataset(split="train", max_items=10):
    """Load and return a list of PIL Images from the HuggingFace 'pokemon' dataset."""
    ds = load_dataset("fofr/pokemons", split="train")
img = ds[0]["image"]
img.show()    dataset = load_dataset("lambdalabs/pokemon-blip-captions", split=split)
    images = []
    for example in dataset.select(range(min(max_items, len(dataset)))):
        image = Image.open(BytesIO(example["image"])).convert("RGB")
        images.append(image)
    return images


def show_images(images, cols=5):
    """Display a list of PIL images in a grid using matplotlib."""
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i])
            ax.axis("off")
        else:
            ax.remove()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    images = load_pokemon_dataset()
    show_images(images)