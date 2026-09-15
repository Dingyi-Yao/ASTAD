from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Optional, Tuple


class Range(NamedTuple):
    start: int
    end: int

@dataclass
class RunConfig:
    # Style image path
    style_image_path: Path = Path("/root/shared-nvme/data/style_grass")
    # Content image path
    content_image_path: Path = Path("/root/shared-nvme/data/GTA")
    # Output path
    output_path: Path = Path('/root/shared-nvme/output_grass')
    # Path to save the inverted latent codes
    latents_path: Path = Path("/root/shared-nvme/data/latents")
    # Path to save the segmentation prior
    seg_prior_path: Path = Path("/root/shared-nvme/seg_prior_vislatent_asy")
    # Random seed
    seed: int = 42
    # Input prompt for inversion
    prompt: Optional[str] = ""
    # Number of timesteps
    num_timesteps: int = 50
    # Number of steps to skip in the denoising process
    skip_steps: int = 30
    # Whether to load previously saved inverted latent codes
    load_latents: bool = True
    
    # Timesteps to apply cross-attention on 64x128 layers
    cross_attn_64_range: Range = Range(start=5, end=20)
    # Timesteps to apply cross-attention on 32x64 layers
    cross_attn_32_range: Range = Range(start=5, end=20)
    # Swap guidance scale
    swap_guidance_scale: float = 1
    # Attention contrasting strength
    contrast_strength: float = 1.67
    # Style transfer name: "AST"
    name: str = "AST"

    # Apply the filtering operation
    filtering: bool = True
    # Apply the cross-attention operation introduced by cross-image attention [Alaluf et al. (2024)]
    cross_attention: bool = True
    # Apply AdaIN per class
    adain_class: bool = True
    # Timesteps to apply class AdaIn step 5 to 19
    class_adain_range: Range = Range(start=5, end=20)
    # Scale for AdaIN fusion
    adain_scale: float = 8.0
    
    # Confidence threshold for pseudo segmentation
    confidence_threshold: float = 0.5
    
    # Lambda for attention thresholding
    threshold_lambda: float = 2.0

    # part of the GTA5 dataset to consider, from 1 to 5000
    img_range: Optional[Tuple[int, ...]] = tuple(range(1, 5001))

    # Number of content images to use for segmentation prior
    nb_content_img_for_seg_prior: int = 100

    # Whether to update the style segmentation by voting
    use_voting: bool = True

    def __post_init__(self):
        self.latents_path = self.latents_path / f"latents_{self.num_timesteps}"
        Path(self.latents_path / "style").mkdir(parents=True, exist_ok=True)
        Path(self.latents_path / "content").mkdir(parents=True, exist_ok=True)
        Path(self.seg_prior_path).mkdir(parents=True, exist_ok=True)
        for sub_dir in ("content_features", "style_features"):
            Path(self.seg_prior_path / sub_dir).mkdir(parents=True, exist_ok=True)

    def update_latents_path(self, name_content_image, name_style_image):
        # Define the paths to store the inverted latents to
        self.style_latent_save_path = self.latents_path / "style" / f"{name_style_image}.pt"
        self.content_latent_save_path = self.latents_path / "content" / f"{name_content_image}.pt"
        