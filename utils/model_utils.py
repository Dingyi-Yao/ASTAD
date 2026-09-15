import torch
from diffusers import DDIMScheduler

from models.stable_diffusion import CrossImageAttentionStableDiffusionPipeline
from models.unet_2d_condition import FreeUUNet2DConditionModel


def get_stable_diffusion_model() -> CrossImageAttentionStableDiffusionPipeline:
    print("Loading Stable Diffusion model...")
    local_model_path = "/root/shared-nvme/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
    device = torch.device(f'cuda') if torch.cuda.is_available() else torch.device('cpu')
    pipe = CrossImageAttentionStableDiffusionPipeline.from_pretrained(local_model_path,
                                                                      safety_checker=None).to(device)
    pipe.unet = FreeUUNet2DConditionModel.from_pretrained(local_model_path, subfolder="unet").to(device)
    pipe.scheduler = DDIMScheduler.from_config(local_model_path, subfolder="scheduler")
    print("Done.")
    return pipe
