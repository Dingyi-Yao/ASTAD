from typing import Optional, Callable

import math
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from config import RunConfig
from constants import *
from models.stable_diffusion import CrossImageAttentionStableDiffusionPipeline
from utils import attention_utils
from utils.adain import adain, adain_class_ratio_adaptive
from utils.model_utils import get_stable_diffusion_model
from utils.seg_utils import generate_pseudo_segmentation
from utils.seg_prior_utils import render_class_tensor
class ASTModel:


    def __init__(self, config: RunConfig, pipe: Optional[CrossImageAttentionStableDiffusionPipeline] = None):
        self.config = config
        self.pipe = get_stable_diffusion_model() if pipe is None else pipe
        self.register_attention_control()

        self.latents_style, self.latents_content = None, None
        self.zs_style, self.zs_content = None, None

        self.label_content = None
        self.label_style = []
        self.feat_style = []

        self.label_content_adain = None
        self.style_mu, self.style_sigma = None, None
        self.onehot_mask_content, self.onehot_mask_style = None, None
        # Style-side pseudo segmentation derived from features for filtering
        self.style_segmentation = None
        self.style_segmentation_prior = None
        self.unique_content_classes = None
        self.enable_edit = False
        self.step = 0

        # For voting-based pseudo segmentation
        self.current_timestep = -1
        self.pseudo_maps_accumulator = []

        # Number of attention layers at 8192 resolution (64x128) in the 'up' blocks
        self.num_attention_layers_8192 = 3

    def set_onehot_masks(self):
        """
        Generates class-based one-hot segmentation masks for content labels.
        """
        n_classes = 20
        onehot_mask_content = np.zeros((n_classes, *self.label_content.shape), dtype=np.uint8)

        for i in range(n_classes):
            onehot_mask_content[i] = (self.label_content == i).astype(np.uint8)

        # Handle undefined class (label 255)
        onehot_mask_content[-1] = (self.label_content == 255).astype(np.uint8)

        self.onehot_mask_content = torch.from_numpy(onehot_mask_content).view(1, 1, n_classes, 512, 1024)
        self.unique_content_classes = torch.from_numpy(np.unique(self.label_content))
        

    # Latents setter
    def set_latents(self, latents_style: torch.Tensor, latents_content: torch.Tensor):
        self.latents_style = latents_style
        self.latents_content = latents_content

    # Noise setter
    def set_noise(self, zs_style: torch.Tensor, zs_content: torch.Tensor):
        self.zs_style = zs_style
        self.zs_content = zs_content

    def get_adain_callback(self) -> Callable:
        """
        Returns a callback function for AdaIN or class-AdaIN based on the current step and config.
        """
        def callback(st: int, t: int, latents: torch.FloatTensor) -> None:
            self.step = st

            if self.config.class_adain_range.start <= self.step <= self.config.class_adain_range.end and self.config.adain_class:
                # Apply class-wise AdaIN using the style segmentation
                adain_out = adain_class_ratio_adaptive(latents[0], latents[1], self.label_content_adain, self.style_segmentation, scale=self.config.adain_scale)
                latents[0] = adain_out
            elif self.step > self.config.class_adain_range.end and self.config.adain_class:
                self.style_segmentation = None
            else:
                # Apply standard AdaIN
                latents[0] = adain(latents[0], latents[1])
                # Clear segmentation if not in class AdaIN range to avoid stale data
                self.style_segmentation = None
        
        return callback

    def update_style_segmentation_by_voting(self):
        """
        Updates the style segmentation by voting on the accumulated pseudo-maps.
        
        This function stacks the collected pseudo-maps, determines the most frequent class
        at each pixel, and updates the style segmentation. If a prior is available, it
        incorporates the prior information to refine the segmentation.
        """
        # Stack all collected maps: (num_layers, H, W)
        stacked_maps = torch.stack(self.pseudo_maps_accumulator)
        # Vote for the most frequent class at each pixel: (H, W)
        voted_map, _ = torch.mode(stacked_maps, dim=0)

        if self.style_segmentation_prior is not None:
            prior = self.style_segmentation_prior
            # only update pixels where all pseudo-maps agree and differ from prior
            all_agree = (stacked_maps == voted_map.unsqueeze(0)).all(dim=0)
            # check if voted_map class exists in content classes
            unique_content_classes = self.unique_content_classes.to(prior.device)
            prior_in_content = torch.isin(prior, unique_content_classes)
            # only update where voted_map differs from prior and prior class exists in content
            # and voted_map is not ignore index 255
            update_mask = all_agree & (voted_map != prior) & (voted_map != 255) & prior_in_content

            new_pseudo_map = prior.clone()
            new_pseudo_map[update_mask] = voted_map[update_mask]
            self.style_segmentation = new_pseudo_map

        else:
            # If no prior is available, use the voted map directly
            self.style_segmentation = voted_map

    def register_attention_control(self):
        """
        Registers a custom attention control mechanism that modifies cross-attention maps
        by selectively applying cross-attention based on feature similarity.
        """
        model_self = self

        class AttentionProcessor:
            def __init__(self, place_in_unet: str):
                self.place_in_unet = place_in_unet

                if not hasattr(F, "scaled_dot_product_attention"):
                    raise ImportError(
                        "AttentionProcessor requires torch 2.0+. Please upgrade your torch installation."
                    )

            def class_consistent_attention_filtering(self, model_self: ASTModel, attn_map, V):
                """
                Filters the attention map to ensure class consistency between content and style features.
                """
                # Determine attention map resolution
                attn_h = int(np.sqrt(attn_map.shape[2] // 2))
                attn_w = attn_h * 2

                style_segmentation = model_self.style_segmentation if model_self.style_segmentation is not None else model_self.style_segmentation_prior
                if model_self.onehot_mask_content is not None:
                    device = attn_map.device

                    content_masks = model_self.onehot_mask_content.to(device=device, dtype=torch.long)
                    n_classes = content_masks.shape[2]
                    content_masks = content_masks.view(1, n_classes, content_masks.shape[-2], content_masks.shape[-1])
                    content_masks = F.interpolate(content_masks.float(), size=(attn_h, attn_w), mode='nearest').long().squeeze(0)

                    content_labels = torch.argmax(content_masks[:-1], dim=0).long()
                    undefined_mask = content_masks[-1].bool()
                    content_labels[undefined_mask] = 255
                    content_labels = content_labels.view(-1)

                    style_labels = style_segmentation.to(device=device, dtype=torch.long)
                    style_labels = F.interpolate(style_labels.unsqueeze(0).unsqueeze(0).float(), size=(attn_h, attn_w), mode='nearest').squeeze().long()
                    style_labels = style_labels.view(-1)

                    valid_content = content_labels != 255
                    valid_style = style_labels != 255

                    # compare using broadcasting
                    same_class_mask = (
                        content_labels.unsqueeze(1) == style_labels.unsqueeze(0)
                    ) & valid_content.unsqueeze(1) & valid_style.unsqueeze(0)

                    attn_scores = attn_map[OUT_INDEX].mean(dim=0)

                    median_val = attn_scores.median()
                    mad = torch.abs(attn_scores - median_val).median()
                    threshold_final = median_val + model_self.config.threshold_lambda * mad
                    high_attention_mask = attn_scores > threshold_final
                    candidate_mask = same_class_mask & high_attention_mask
                    return candidate_mask

                return None, None
            
            def __call__(self,
                         attn,
                         hidden_states: torch.Tensor,
                         encoder_hidden_states: Optional[torch.Tensor] = None,
                         attention_mask=None,
                         temb=None,
                         perform_swap: bool = False):
                
                residual = hidden_states

                if attn.spatial_norm is not None:
                    hidden_states = attn.spatial_norm(hidden_states, temb)

                input_ndim = hidden_states.ndim
                
                if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

                batch_size, sequence_length, _ = (
                    hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                )

                if attention_mask is not None:
                    attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                    attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

                if attn.group_norm is not None:
                    hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                # Check if a new timestep has started
                if model_self.step != model_self.current_timestep:
                    model_self.current_timestep = model_self.step
                    model_self.pseudo_maps_accumulator = []

                    if not model_self.config.use_voting:
                        model_self.style_segmentation = model_self.style_segmentation_prior

                # Generate and vote on pseudo-maps only at the 8192 resolution in 'up' blocks
                is_target_res = sequence_length == 8192
                is_cross = encoder_hidden_states is not None

                if perform_swap and not is_cross and is_target_res and "up" in self.place_in_unet and model_self.enable_edit and model_self.config.use_voting:
                    # Generate a new pseudo-map
                    pseudo_map = generate_pseudo_segmentation(hidden_states, model_self.label_content, confidence_threshold=model_self.config.confidence_threshold)
                    
                    model_self.pseudo_maps_accumulator.append(pseudo_map)
                    # Vote
                    # If enough maps are collected, perform voting
                    if len(model_self.pseudo_maps_accumulator) >= model_self.num_attention_layers_8192:
                        model_self.update_style_segmentation_by_voting()

                query = attn.to_q(hidden_states)
                
                if not is_cross:
                    encoder_hidden_states = hidden_states
                elif attn.norm_cross:
                    encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

                key = attn.to_k(encoder_hidden_states)
                value = attn.to_v(encoder_hidden_states)
                
                inner_dim = key.shape[-1]
                head_dim = inner_dim // attn.heads
                should_mix = False
                
                # Potentially apply cross image attention operation
                # To do so, we need to be in a self-attention layer in the decoder part of the denoising network
                if model_self.config.cross_attention:
                    if perform_swap and not is_cross and "up" in self.place_in_unet and model_self.enable_edit:
                        if attention_utils.should_mix_keys_and_values(model_self, hidden_states):
                            should_mix = True
                            # Inject the appearance's keys and values
                            key[OUT_INDEX] = key[STYLE_INDEX]
                            value[OUT_INDEX] = value[STYLE_INDEX]

                query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                
                # Check directly if we are in the filtering scenario to enable optimization
                do_filtering = model_self.config.filtering and perform_swap and not is_cross and "up" in self.place_in_unet and model_self.enable_edit and should_mix

                # Compute the cross attention and apply contrasting operation
                hidden_states, attn_weight = attention_utils.compute_scaled_dot_product_attention(
                    query, key, value,
                    edit_map=perform_swap and model_self.enable_edit and should_mix,
                    is_cross=is_cross,
                    contrast_strength=model_self.config.contrast_strength,
                    output_attn_only=do_filtering,
                )
                
                # Apply the filtering operation
                if do_filtering:
                    filtering_mask = self.class_consistent_attention_filtering(model_self, attn_weight, value)
                    if filtering_mask is not None:
                        out_content = attn_weight[CONTENT_INDEX] @ value[CONTENT_INDEX]
                        
                        mask_expanded = filtering_mask.unsqueeze(0).to(dtype=attn_weight.dtype)
                        masked_attn_weight = attn_weight[OUT_INDEX] * mask_expanded
                        out_raw = masked_attn_weight @ value[OUT_INDEX]
                        
                        # Calculate mixing factor from the masked attention weights
                        mix_factor = masked_attn_weight.sum(dim=-1, keepdim=True)
                        out_combined = out_raw + (1 - mix_factor) * out_content

                        # Reconstruct hidden_states since we skipped the full batch matmul
                        final_states = []
                        for i in range(attn_weight.shape[0]):
                            if i == OUT_INDEX:
                                final_states.append(out_combined)
                                
                            elif i == CONTENT_INDEX:
                                final_states.append(out_content)
                            else:
                                final_states.append(attn_weight[i] @ value[i])
                        hidden_states = torch.stack(final_states)
                    else:
                        # Filtering failed, fallback to standard computation
                        hidden_states = attn_weight @ value

                hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
                hidden_states = hidden_states.to(query[OUT_INDEX].dtype)

                # linear proj
                hidden_states = attn.to_out[0](hidden_states)
                # dropout
                hidden_states = attn.to_out[1](hidden_states)

                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

                if attn.residual_connection:
                    hidden_states = hidden_states + residual

                hidden_states = hidden_states / attn.rescale_output_factor
                
                return hidden_states

        def register_recr(net_, count, place_in_unet):
            if net_.__class__.__name__ == 'ResnetBlock2D':
                pass
            if net_.__class__.__name__ == 'Attention':
                net_.set_processor(AttentionProcessor(place_in_unet + f"_{count + 1}"))
                return count + 1
            elif hasattr(net_, 'children'):
                for net__ in net_.children():
                    count = register_recr(net__, count, place_in_unet)
            return count

        cross_att_count = 0
        sub_nets = self.pipe.unet.named_children()
        for net in sub_nets:
            if "down" in net[0]:
                cross_att_count += register_recr(net[1], 0, "down")
            elif "up" in net[0]:
                cross_att_count += register_recr(net[1], 0, "up")
            elif "mid" in net[0]:
                cross_att_count += register_recr(net[1], 0, "mid")
