from pathlib import Path
from typing import Sequence, Tuple, Union


import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import time
import palettes

from constants import CONTENT_INDEX, STYLE_INDEX
from .seg_prior_utils import aggregate_region_prototypes, match_regions, render_class_tensor


def generate_pseudo_segmentation(
    hidden_states: torch.Tensor,
    label_content: np.ndarray,
    num_classes: int = 20,
    ignore_index: int = 255,
    confidence_threshold: float = 0.4
) -> torch.Tensor:
    """
    Generates a pseudo-segmentation map for the style image based on feature similarity
    to class centroids from the content image.

    Args:
        hidden_states (torch.Tensor): The feature maps from the UNet, shape (B, H*W, C),
                                      where B should be at least 2 (content + style).
        label_content (np.ndarray): The ground truth segmentation map for the content image,
                                          shape (H_orig, W_orig).
        num_classes (int): The number of segmentation classes.
        ignore_index (int): The ignore index in the label map.

    Returns:
        torch.Tensor: The pseudo-segmentation map for the style image, shape (H_orig, W_orig).
    """
    orig_h, orig_w = label_content.shape
    label_content = torch.from_numpy(label_content).long()
    
    hidden_states = F.normalize(hidden_states, p=2, dim=-1)
    
    feat_c = hidden_states.shape[-1]
    # H=64, W=128
    feat_h = int((hidden_states.shape[1] // 2) ** 0.5)
    feat_w = feat_h * 2

    # (B, H*W, C) -> (C, H, W)
    content_features = hidden_states[CONTENT_INDEX].T.view(feat_c, feat_h, feat_w)
    style_features = hidden_states[STYLE_INDEX].T.view(feat_c, feat_h, feat_w)

    # Aggregate prototypes from content features and labels
    region_info = aggregate_region_prototypes(
        features=[content_features],
        seg_map=[label_content],
        num_classes=num_classes,
        ignore_index=ignore_index
    )

    # Match style features to prototypes
    patch_probs = match_regions(style_features, region_info)

    # Upsample probabilities and get final segmentation map
    # (H_p, W_p, num_classes) -> (1, num_classes, H_p, W_p)
    prob_tensor = patch_probs.permute(2, 0, 1).unsqueeze(0)
    prob_tensor = F.interpolate(prob_tensor, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
    
    max_probs, seg_map = prob_tensor.max(dim=1)
    seg_map = seg_map.squeeze(0).long()
    

    mask = max_probs.squeeze(0) < confidence_threshold
    seg_map[mask] = ignore_index

    return seg_map

