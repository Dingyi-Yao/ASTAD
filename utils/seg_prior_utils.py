from pathlib import Path
from typing import Sequence, Tuple, Union


import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


from utils.image_utils import load_size
from utils.mask_utils import relabel_image
import palettes


DINOV2_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
DINOV2_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _feature_cache_path(base_path: Path, image_path: Path, kind: str) -> Path:
    """
    Return the cache path for DINO features of a given image.
    
    Args:
        base_path: Base directory for caching.
        image_path: Path to the original image.
        kind: 'content' or 'style' indicating the type of features.

    Returns:
        Path to the cached feature tensor.
    """
    path_obj = Path(image_path)
    cache_dir = base_path / f"{kind}_features"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{path_obj.stem}.pt"


def _load_preprocessed_tensor(image_path: Path, size: int = 518) -> torch.Tensor:
    """Load image, resize to a specific size, and convert to a normalized tensor."""
    resized = load_size(image_path, size=size)
    if isinstance(resized, np.ndarray):
        image_np = resized
    else:
        image_np = np.array(resized.convert('RGB'))
    tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    return tensor


def _load_or_compute_dino_features(image_path: Path, cfg, kind: str, device: torch.device) -> torch.Tensor:
    """
    Load cached DINO features if available, otherwise compute and cache them.
    
    Args:
        image_path: Path to the input image.
        cfg: Configuration object with seg_prior_path attribute.
        kind: 'content' or 'style' indicating the type of features.
        device: Target device for computation.

    Returns:
        Tensor of DINO features.
    """
    cache_path = _feature_cache_path(cfg.seg_prior_path, image_path, kind)
    if cache_path.exists():
        print(f"Loading cached {kind} features from {cache_path}")
        return torch.load(cache_path, map_location=device)

    features = extract_dino_features(_load_preprocessed_tensor(image_path), device)
    torch.save(features.cpu(), cache_path)
    return features



def _get_dinov2_model(device) -> torch.nn.Module:
    """Load the DINOv2 model from a local directory."""
    # Path to the local DINOv2 repository
    dinov2_repo_path = '/root/shared-nvme/facebookresearch-dinov2-b8931f7'
    
    # Load model structure from the local repository
    model = torch.hub.load(dinov2_repo_path, 'dinov2_vits14', source='local')
    
    # Load weights from the local .pth file
    local_weights_path = '/root/shared-nvme/dino/dinov2_vits14_pretrain.pth'
    state_dict = torch.load(local_weights_path, map_location='cpu')
    model.load_state_dict(state_dict)
    
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model.to(device)



@torch.no_grad()
def extract_dino_features(image_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Extract dense patch-level features from a preprocessed image tensor using DINOv2.

    Args:
        image_tensor: Tensor of shape (C, H, W) in range [0, 1].
        device: Target device for inference.

    Returns:
        Tensor of shape (C_feat, H_p, W_p) containing L2-normalized patch features.
    """
    if image_tensor.dim() != 3 or image_tensor.shape[0] != 3:
        raise ValueError("image_tensor must be shaped as (3, H, W)")

    model = _get_dinov2_model(device)

    mean = DINOV2_MEAN.to(device=device, dtype=image_tensor.dtype)
    std = DINOV2_STD.to(device=device, dtype=image_tensor.dtype)

    input_tensor = image_tensor.unsqueeze(0).to(device=device, dtype=torch.float32)
    input_tensor = (input_tensor - mean) / std

    features = model.forward_features(input_tensor)

    patch_tokens = features.get('x_norm_patchtokens', None)

    _, _, feat_dim = patch_tokens.shape

    patch_embed = getattr(model, 'patch_embed', None)
    patch_size = getattr(patch_embed, 'patch_size', None)

    if isinstance(patch_size, (tuple, list)):
        patch_h_size, patch_w_size = patch_size
    else:
        patch_h_size = patch_w_size = patch_size

    height_patches = input_tensor.shape[-2] // patch_h_size
    width_patches = input_tensor.shape[-1] // patch_w_size

    features = patch_tokens.view(1, height_patches, width_patches, feat_dim)
    features = features.permute(0, 3, 1, 2).contiguous().squeeze(0)
    features = F.normalize(features, p=2, dim=0)

    return features


@torch.no_grad()
def aggregate_region_prototypes(
    features: Union[torch.Tensor, Sequence[torch.Tensor]],
    seg_map: Union[torch.Tensor, Sequence[torch.Tensor]],
    num_classes: int = 20,
    ignore_index: int = 255
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Aggregate patch-level features into region prototypes based on segmentation labels.

    Args:
        features: Tensor (C_feat, H_p, W_p) or sequence of such tensors containing normalized features.
        seg_map: Tensor (H, W) or sequence of tensors with integer labels aligned to ``features``.
        num_classes: Total number of Cityscapes classes including the ignore slot.
        ignore_index: Label value to treat as background/ignore.

    Returns:
        prototypes: Tensor (num_classes, C_feat) containing class prototypes.
        valid_mask: Bool tensor (num_classes,) indicating which prototypes are valid.
    """
    if isinstance(features, torch.Tensor):
        features_list = [features]
    else:
        features_list = list(features)

    if isinstance(seg_map, torch.Tensor):
        seg_map_list = [seg_map]
    else:
        seg_map_list = list(seg_map)


    if len(features_list) != len(seg_map_list):
        print("features_list length:", len(features_list))
        print("seg_map_list length:", len(seg_map_list))
        raise ValueError("features and seg_map must contain the same number of elements")

    if not features_list:
        raise ValueError("features must contain at least one tensor")

    first_feature = features_list[0]
    if first_feature.dim() != 3:
        raise ValueError("each feature tensor must be 3D (C_feat, H_p, W_p)")

    device = first_feature.device
    feat_dim = first_feature.shape[0]

    proto_sums = torch.zeros(num_classes, feat_dim, dtype=first_feature.dtype, device=device)
    counts = torch.zeros(num_classes, dtype=first_feature.dtype, device=device)

    for feat, seg in zip(features_list, seg_map_list):
        if feat.dim() != 3:
            raise ValueError("each feature tensor must be 3D (C_feat, H_p, W_p)")

        if feat.shape[0] != feat_dim:
            raise ValueError("all feature tensors must share the same channel dimension")

        if seg.dim() != 2:
            raise ValueError("each seg_map tensor must be 2D")

        if feat.device != device:
            feat = feat.to(device)

        seg = seg.to(device=device, dtype=torch.long)

        features_upsampled = F.interpolate(
            feat.unsqueeze(0),
            size=seg.shape,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        ignore_tensor = torch.full_like(seg, num_classes - 1)
        seg_processed = torch.where(seg == ignore_index, ignore_tensor, seg)

        flat_features = features_upsampled.view(features_upsampled.shape[0], -1).permute(1, 0)
        flat_features = F.normalize(flat_features, p=2, dim=1)
        flat_labels = seg_processed.view(-1)

        for class_idx in range(num_classes - 1):
            class_mask = flat_labels == class_idx
            if not torch.any(class_mask):
                continue

            class_features = flat_features[class_mask]
            proto_sums[class_idx] += class_features.sum(dim=0)
            counts[class_idx] += class_features.shape[0]

    prototypes = torch.zeros_like(proto_sums)
    valid_mask = counts > 0
    if torch.any(valid_mask):
        averaged = proto_sums[valid_mask] / counts[valid_mask].unsqueeze(1)
        prototypes[valid_mask] = F.normalize(averaged, p=2, dim=1)

    return prototypes, valid_mask


@torch.no_grad()
def match_regions(features: torch.Tensor, region_info: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """
    Match style-image features to content region prototypes to obtain class probabilities.

    Args:
        features: Tensor (C_feat, H_p, W_p) of normalized features.
        region_info: Tuple containing prototypes tensor and valid mask tensor.

    Returns:
        Tensor (H_p, W_p, num_classes) with per-patch probability distributions.
    """
    prototypes, valid_mask = region_info
    num_classes, _ = prototypes.shape
    c_feat, height_p, width_p = features.shape

    flat_features = features.view(c_feat, -1).permute(1, 0)
    flat_features = F.normalize(flat_features, p=2, dim=1)

    valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)

    valid_prototypes = prototypes[valid_mask]
    valid_prototypes = F.normalize(valid_prototypes, p=2, dim=1)

    similarities = flat_features @ valid_prototypes.t()
    # no softmax here to keep raw similarity scores
    probs = torch.zeros(flat_features.shape[0], num_classes, device=features.device)
    probs[:, valid_indices] = similarities

    return probs.view(height_p, width_p, num_classes)


@torch.no_grad()
def compute_style_segmentation_prior(
    cfg,
    path_content_img_list,
    path_content_label_list,
    path_style_img
) -> torch.Tensor:
    """
    Generate a style-image pseudo segmentation prior based on DINOv2 feature matching.

    Args:
        model: Active model instance used to infer the target device.
        path_content_img_list: Iterable of paths to the content RGB images.
        path_content_label_list: Iterable of paths to the content segmentation label images (aligned with ``path_content_img_list``).
        path_style_img: Path to the style RGB image.

    Returns:
        Tensor (H_orig, W_orig) containing integer class predictions (num_classes - 1 marks ignore).
    """
    device = torch.device('cuda')

    path_content_img_list = list(path_content_img_list)
    path_content_label_list = list(path_content_label_list)

    if len(path_content_img_list) != len(path_content_label_list):
        raise ValueError("path_content_img_list and path_content_label_list must have the same length")

    if not path_content_img_list:
        raise ValueError("path_content_img_list must contain at least one element")

    content_features_list = []
    content_label_tensors = []

    for img_path, label_path in zip(path_content_img_list, path_content_label_list):
        content_features = _load_or_compute_dino_features(Path(img_path), cfg, 'content', device)
        label_image = Image.open(label_path)
        label_resized = load_size(label_image, size=512)
        if not isinstance(label_resized, Image.Image):
            label_resized = Image.fromarray(label_resized)

        content_label = relabel_image(label_resized, palettes.GTA)
        content_label_tensor = torch.from_numpy(content_label).long().to(device)

        content_features_list.append(content_features)
        content_label_tensors.append(content_label_tensor)

    style_features = _load_or_compute_dino_features(Path(path_style_img), cfg, 'style', device)

    region_info = aggregate_region_prototypes(content_features_list, content_label_tensors)
    patch_probs = match_regions(style_features, region_info)

    target_height, target_width = content_label_tensors[0].shape

    prob_tensor = patch_probs.permute(2, 0, 1).unsqueeze(0)
    prob_tensor = F.interpolate(
        prob_tensor,
        size=(target_height, target_width),
        mode='bilinear',
        align_corners=False,
    )
    class_tensor = prob_tensor.argmax(dim=1).squeeze(0).long()
    
    return class_tensor



def render_class_tensor(class_tensor: torch.Tensor, save_path: Union[str, Path] = None) -> Image.Image:
    class_np = class_tensor.detach().cpu().numpy().astype(np.uint8)
    pil_img = Image.fromarray(class_np, mode='P')

    palette = np.array(palettes.CITYSCAPES, dtype=np.uint8)
    if palette.ndim != 2 or palette.shape[1] != 3:
        raise ValueError("palettes.GTA must be of shape (N, 3)")

    # Ensure palette has 256 entries
    if palette.shape[0] < 256:
        padding = np.zeros((256 - palette.shape[0], 3), dtype=np.uint8)
        palette = np.vstack((palette, padding))
    
    palette = palette[:256]
    
    # Explicitly set index 255 (ignore_index) to black
    palette[255] = [0, 0, 0]

    palette_flat = palette.flatten()
    pil_img.putpalette(palette_flat.tolist())

    pil_rgb = pil_img.convert('RGB')
    if save_path is not None:
        pil_rgb.save(str(save_path))
    return pil_rgb