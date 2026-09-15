import os
import sys
import random
import json

import numpy as np
import pyrallis
import torch
from PIL import Image
from tqdm import tqdm
from diffusers.training_utils import set_seed

sys.path.append(".")
sys.path.append("..")

from ast_model import ASTModel
from utils.latent_utils import load_or_invert_one_image, get_init_latents_and_noises
from utils.mask_utils import *
from utils.seg_prior_utils import compute_style_segmentation_prior
import palettes

# change config path here
from config import RunConfig, Range


@pyrallis.wrap()
def main(cfg: RunConfig):
    """
    Main function. Calls the function to transfer style to either a dataset of images or a single image.
    """
    random.seed(0)
    cfg.output_path = cfg.output_path / cfg.name / f"ts{cfg.num_timesteps}_skip{cfg.skip_steps}"
    cfg.output_path.mkdir(parents=True, exist_ok=True)
    # Transfer style to a dataset
    transfer_dataset(cfg=cfg, sampling='ordered-repeat')

def compute_prior(
    cfg: RunConfig,
    model: ASTModel,
    img_style: str = None,
    available_content: list = None
):
    """Compute style segmentation prior using sampled content-label pairs and attach to the model."""

    content_image_root = cfg.content_image_path / "images"
    content_label_root = cfg.content_image_path / "labels"

    if cfg.nb_content_img_for_seg_prior > len(available_content):
        raise ValueError("Number of content images for segmentation prior exceeds available images.")
    
    sampled_pairs = []
    sampled_numbers = random.sample(available_content, cfg.nb_content_img_for_seg_prior)
    
    sampled_numbers.sort()
    for content_img_number in sampled_numbers:
        img_name = f"{content_img_number:0>5d}.png"
        path_content_img = content_image_root / img_name
        path_content_label = content_label_root / img_name
        sampled_pairs.append((path_content_img, path_content_label))
        

    style_segmentation_prior = compute_style_segmentation_prior(
        cfg=cfg,
        path_content_img_list=[pair[0] for pair in sampled_pairs],
        path_content_label_list=[pair[1] for pair in sampled_pairs],
        path_style_img= cfg.style_image_path / "images" / img_style
    )
    model.style_segmentation_prior = style_segmentation_prior

def transfer_dataset(cfg: RunConfig, sampling='ordered-repeat'):
    """
    Transfers style to a dataset of content images. Selects images based on a sampling strategy.
    Args:
        cfg: Model configuration.
        sampling: Sampling strategy ('rcs', 'ordered', 'ordered-repeat' or 'uniform').
    """
    style_image_root = cfg.style_image_path / "images"
    style_label_root = cfg.style_image_path / "labels"

    set_seed(cfg.seed)
    model = ASTModel(cfg)
    model.pipe.scheduler.set_timesteps(cfg.num_timesteps)
    model.config = cfg

    # Load style labels
    list_style_images = os.listdir(style_image_root)
    # list_style_labels, _ = open_style_labels(list_style_labels, style_label_root, palettes.CITYSCAPES)

    # Determine available content images
    if cfg.img_range is not None:
        available_content = cfg.img_range
    else:
        # Load all images from content folder
        content_files = sorted(os.listdir(cfg.content_image_path / "images"))
        available_content = []
        for f in content_files:
            if f.endswith('.png') or f.endswith('.jpg'):
                available_content.append(int(f.split('.')[0]))

    if sampling == 'ordered-repeat':
        # Ordered sampling with the same images for each style
        selection = []
        if len(available_content) > 0:
            while len(selection) < len(available_content):
                selection.extend(available_content)
            selection = selection[:len(available_content)]
        else:
            selection = []
        samples_content = selection * len(list_style_images)
    else:
        raise NotImplementedError(f"Sampling strategy '{sampling}' not implemented.")
        
    it = iter(samples_content)
    output_data = cfg.output_path

    for img_style in tqdm(list_style_images):
        cfg.output_path = output_data / img_style.split("_leftImg8bit.png")[0]
        cfg.output_path.mkdir(parents=True, exist_ok=True)
        
        compute_prior(
                cfg=cfg,
                model=model,
                img_style=img_style,
                available_content=available_content
            )
        
        for i in tqdm(range(len(available_content))):
            img_number = next(it)

            
            img_name = f"{img_number:0>5d}.png"
            path_content_img = cfg.content_image_path / "images" / img_name
            path_content_label = cfg.content_image_path / "labels" / img_name

            # Handle different style naming conventions
            name_close_style = img_style.split("_leftImg8bit.png")[0]
            path_style_img = style_image_root / f'{name_close_style}_leftImg8bit.png'
            if not path_style_img.exists():
                path_style_img = style_image_root / f'{name_close_style}_rgb_anon.png'


            # Perform style transfer - style label path is now None since not needed
            load_latents_couple(cfg, model, path_content_img, path_content_label, path_style_img, None, img_number)

    # Save configuration
    pyrallis.dump(cfg, open(cfg.output_path / 'config.yaml', 'w'))


def load_latents_couple(cfg: RunConfig, model: ASTModel, path_content_img, path_content_label, path_style_img, path_style_label, img_number=3):
    """
    Loads and processes latents for both content and style images.
    Note: style label is no longer processed - only content label is used.
    """
    with torch.no_grad():
        # Process labels for content only - no style label needed
        model.label_content, model.label_content_adain = process_label(path_content_label, palettes.GTA)
        
        # Load or invert images to latents
        cfg.update_latents_path(path_content_img.stem, path_style_img.stem)
        latents_style, noise_style = load_or_invert_one_image(model.pipe, cfg, img_path=path_style_img, type_img="style")
        latents_content, noise_content = load_or_invert_one_image(model.pipe, cfg, img_path=path_content_img, type_img="content")
        model.set_latents(latents_style, latents_content)
        model.set_noise(noise_style, noise_content)
        model.set_onehot_masks()

        # Run the style transfer process
        run_style_transfer(model=model, cfg=cfg, img_number=img_number)


def run_style_transfer(model: ASTModel, cfg: RunConfig, img_number: str):
    """
    Setups and runs the diffusion process.
    """
    init_latents, init_zs = get_init_latents_and_noises(model=model, cfg=cfg)
    model.pipe.scheduler.set_timesteps(cfg.num_timesteps)
    model.enable_edit = True  # Enable cross-image attention layers

    start_step = min(cfg.cross_attn_32_range.start, cfg.cross_attn_64_range.start)
    end_step = max(cfg.cross_attn_32_range.end, cfg.cross_attn_64_range.end)

    # Run diffusion process
    images = model.pipe(
        prompt=[cfg.prompt] * 3,
        latents=init_latents,
        guidance_scale=1.0,
        num_inference_steps=cfg.num_timesteps,
        swap_guidance_scale=cfg.swap_guidance_scale,
        callback=model.get_adain_callback(),
        eta=1,
        zs=init_zs,
        generator=torch.Generator('cuda').manual_seed(cfg.seed),
        cross_image_attention_range=Range(start=start_step, end=end_step)
    ).images

    # Save outputs
    (cfg.output_path / "transfer").mkdir(parents=True, exist_ok=True)
    # (cfg.output_path / "input").mkdir(parents=True, exist_ok=True)
    # (cfg.output_path / "joined").mkdir(parents=True, exist_ok=True)

    images[0].save(cfg.output_path / "transfer" / f"{cfg.name}_transfer_{img_number}.png")
    # images[1].save(cfg.output_path / "input" / f"{cfg.name}_style_{img_number}.png")
    # images[2].save(cfg.output_path / "input" / f"{cfg.name}_content_{img_number}.png")

    return images



# Entry point
if __name__ == '__main__':
    main()
