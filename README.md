# imagegen

This repo supports the post [Image Generative AI - DDPM](https://mrcartoonology.github.io/jekyll/update/2025/06/24/image_gen_ai_ddpm.html). 
It has not been cleaned up for general use but provides reference for the experiments reported on. 

`main` branch is not used for the post - support is in the branches

* `first_run` implement lightweight DDP from scratch, report on unexpected sampling behavior
* `evalddpm` add diagnostic metrics to dig into what is happening
* `fix_run` run Hugging Face code from notebooks 

# first_run support

This code is in the branch
[first_run](https://github.com/MrCartoonology/imagegen/tree/first_run)

## Data
one first needs to download the celeba headshot dataset and put it somewhere - or use your own dataset. There are 202,599 jpg's in the dataset.

## Config
Modify the [config/ddpm_config.yaml](https://github.com/MrCartoonology/imagegen/blob/first_run/config/ddpm_config.yaml) file for the run. 

## Run
from the repo directory, assuming uv is used for package management 
```
uv run python src/imagegen/ddpm_pipeline.py
```
will kick off training, save logs for tensorboard, and the trained model.

# adding metrics
This code is in the branch [evalddpm](https://github.com/MrCartoonology/imagegen/tree/evalddpm) and uses a different pipeline file
```
uv run python src/imagegen/ddpmeval.py
```
it creates images to make screenshots from

## running Hugging Face
this code is in the branch  [fix_run](https://github.com/MrCartoonology/imagegen/tree/fix_run). It contains the notebooks referenced in the post. These start as a copy of the Hugging Face colab from the [Annoted Diffusion Model](https://huggingface.co/blog/annotated-diffusion)



