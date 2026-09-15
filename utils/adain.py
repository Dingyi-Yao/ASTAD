import numpy as np
import torch
from scipy.ndimage import binary_erosion
from PIL import Image
import torch.nn.functional as F
from sklearn.cluster import KMeans, DBSCAN
import numpy as np
from scipy import ndimage
from collections import Counter

torch.manual_seed(42)
np.random.seed(42)


def adain(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def adain_pixel(content_feat, content_mu, content_sigma, style_mu, style_sigma):
    """
    Adaptive Instance Normalization (AdaIN) at pixel level.
    Args:
        content_feat (torch.Tensor): Content features to be stylized [C, H, W]
        content_mu (torch.Tensor): Content mean per pixel [C, H, W]
        content_sigma (torch.Tensor): Content std per pixel [C, H, W]
        style_mu (torch.Tensor): Style mean per pixel [C, H, W]
        style_sigma (torch.Tensor): Style std per pixel [C, H, W]
    Returns:
        torch.Tensor: Stylized content features [C, H, W]
    """
    size = content_feat.size()
    normalized_feat = (content_feat - content_mu.expand(size)) / content_sigma.expand(size)
    return normalized_feat * style_sigma.expand(size) + style_mu.expand(size)


def process_mask(feats, binary_mask, nb_class):
    if isinstance(binary_mask, np.ndarray):
        binary_mask = torch.from_numpy(binary_mask)
    binary_mask = binary_mask.cpu()
    # extract one class from segmentation map
    processed_mask = np.where(binary_mask == nb_class, 1, 0)
    
    processed_mask = binary_erosion(processed_mask)
    # resize mask to match the features
    processed_mask = processed_mask.astype(np.float32)
    
    height = feats.shape[1]
    width = feats.shape[2]
    processed_mask = np.array(Image.fromarray(processed_mask).resize((width, height), resample=Image.NEAREST))

    processed_mask = torch.from_numpy(processed_mask).cuda()
    # >0.5 means : turn to boolean mask
    return feats, processed_mask > 0.5

def adain_class_ratio_adaptive(content_feat, style_feat, content_label, style_segmentation, scale):
    """
    Apply class-wise AdaIN using a provided style segmentation map.

    Args:
        content_feat: Content features to be stylized [C, H, W]
        style_feat: Style features [C, H, W]
        content_label: Content segmentation map [H, W]
        style_segmentation: Style segmentation map [H, W]
        scale: Scale factor for beta map calculation
        
    Returns:
        torch.Tensor: Stylized content features with AdaIN applied [1, C, H, W].
    """

    torch.manual_seed(42)
    np.random.seed(42)

    original_content_feat = content_feat.clone()

    # Initialize style statistics tensors
    style_mu = torch.zeros_like(content_feat)
    style_sigma = torch.ones_like(content_feat)
    
    content_mu = torch.zeros_like(content_feat)
    content_sigma = torch.ones_like(content_feat)
    
    beta_map = torch.full(content_label.shape, 1.0, dtype=torch.float32, device=content_feat.device)

    if isinstance(style_segmentation, torch.Tensor):
        style_segmentation = style_segmentation.cpu().numpy()

    # Process each class to apply style transfer
    for c in range(19):
        # Create masks for the current class for both content and style
        _, content_mask_c = process_mask(content_feat, content_label, c)
        _, style_mask_c = process_mask(style_feat, style_segmentation, c)

        # Skip if class is not present in either content or style
        style_has_content = style_mask_c if isinstance(style_mask_c, bool) else style_mask_c.any()
        if not content_mask_c.any() or not style_has_content:
            continue

        # Calculate mean and std for content features of class c
        content_mu_c, content_sigma_c = calc_mean_std_smooth(content_feat, mask=content_mask_c)
        
        # Calculate mean and std for style features of class c
        style_mu_c, style_sigma_c = calc_mean_std_smooth(style_feat, mask=style_mask_c)

        # If stats are valid, populate the per-pixel stat maps
        if content_mu_c is not None and style_mu_c is not None:
            # mu and sigma (C,1,1) -> (C,H,W)
            content_mu[:, content_mask_c] = content_mu_c.expand_as(content_feat)[:, content_mask_c]
            content_sigma[:, content_mask_c] = content_sigma_c.expand_as(content_feat)[:, content_mask_c]
            style_mu[:, content_mask_c] = style_mu_c.expand_as(content_feat)[:, content_mask_c]
            style_sigma[:, content_mask_c] = style_sigma_c.expand_as(content_feat)[:, content_mask_c]
            
            # Calculate beta based on pixel count ratio
            content_pixel_count = content_mask_c.sum()
            style_pixel_count = style_mask_c.sum()
            if content_pixel_count > 0:
                total_pixels = content_pixel_count + style_pixel_count + 1e-5
                relative_diff = (content_pixel_count - style_pixel_count) / total_pixels
                ratio = torch.sigmoid(scale * relative_diff)
                
                beta_map[content_mask_c] = ratio

    # Apply AdaIN with calculated statistics
    stylized_content_feat = adain_pixel(content_feat, content_mu, content_sigma, style_mu, style_sigma)
    
    # Blend original and stylized features based on beta map
    final_feat = beta_map.unsqueeze(0) * original_content_feat + (1 - beta_map.unsqueeze(0)) * stylized_content_feat
    # final_feat = stylized_content_feat
    final_feat = final_feat.unsqueeze(0)
    return final_feat

def calc_mean_std_smooth(feat, eps=1e-5, mask=None):
    """
    Calculate the mean and standard deviation of features with a mask.
    Args:
        feat (torch.Tensor): Feature tensor of shape (C, H, W).
        eps (float): Small value to avoid division by zero.
        mask (torch.Tensor): Binary mask tensor of shape (H, W).
        Returns:
        Tuple[torch.Tensor, torch.Tensor]: Mean and standard deviation tensors of shape (C, 1, 1).
    """
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 3)
    C = size[0]
    
    mask = mask.view(-1)  # Flatten the mask
    feat_flat = feat.view(C, -1)  # Flatten the feature tensor

    # Compute weighted mean
    weighted_sum = (feat_flat * mask).sum(dim=1)
    sum_of_weights = mask.sum()
    if sum_of_weights.item() == 0:
        return None, None
    weighted_mean = (weighted_sum / sum_of_weights).view(C, 1, 1)

    # Compute weighted variance
    squared_diff = (feat_flat - weighted_mean.view(C, -1)) ** 2
    weighted_variance = (squared_diff * mask).sum(dim=1) / sum_of_weights
    weighted_variance += eps  # Add epsilon for numerical stability
    weighted_std = weighted_variance.sqrt().view(C, 1, 1)

    return weighted_mean, weighted_std

def calc_mean_std(feat, eps=1e-5, mask=None):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    if len(size) == 2:
        return calc_mean_std_2d(feat, eps, mask)
    
    assert (len(size) == 3)
    C = size[0]
    if mask is not None:
        feat_var = feat.view(C, -1)[:, mask.view(-1) == 1].var(dim=1) + eps
        feat_std = feat_var.sqrt().view(C, 1, 1)
        feat_mean = feat.view(C, -1)[:, mask.view(-1) == 1].mean(dim=1).view(C, 1, 1)
    else:
        feat_var = feat.view(C, -1).var(dim=1) + eps
        feat_std = feat_var.sqrt().view(C, 1, 1)
        feat_mean = feat.view(C, -1).mean(dim=1).view(C, 1, 1)

    return feat_mean, feat_std

def calc_mean_std_2d(feat, eps=1e-5, mask=None):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 2)
    C = size[0]
    if mask is not None:
        feat_var = feat.view(C, -1)[:, mask.view(-1) == 1].var(dim=1) + eps
        feat_std = feat_var.sqrt().view(C, 1)
        feat_mean = feat.view(C, -1)[:, mask.view(-1) == 1].mean(dim=1).view(C, 1)
    else:
        feat_var = feat.view(C, -1).var(dim=1) + eps
        feat_std = feat_var.sqrt().view(C, 1)
        feat_mean = feat.view(C, -1).mean(dim=1).view(C, 1)

    return feat_mean, feat_std
