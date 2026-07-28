
import torch
import torch.nn.functional as F
from typing import Any, Sequence, Tuple, Optional
from .types import InterventionConfig


VALID_SIMILARITY_MODES = ("elementwise", "cosine", "gt")


def compute_reference_velocity(
    z_t: torch.Tensor,
    z_0: torch.Tensor,
    sigma: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Recover the velocity that would denoise z_t back to the reference latent."""
    return (z_t - z_0) / (sigma + eps)


def compute_element_similarity(
    v_pred: torch.Tensor,
    v_ref: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Element-wise similarity where 1 means exact match and 0 means large drift."""
    pred_float = v_pred.float()
    ref_float = v_ref.float()

    diff_abs = torch.abs(pred_float - ref_float)

    ref_abs = torch.abs(ref_float) + eps

    similarity = ref_abs / (ref_abs + diff_abs)

    return similarity


def normalize_similarity_mode(mode: Optional[str]) -> str:
    if mode is None:
        return "elementwise"

    normalized = str(mode).strip().lower()
    aliases = {
        "cos": "cosine",
        "cosine_token": "cosine",
        "token_cosine": "cosine",
        "gt_mask": "gt",
        "ground_truth": "gt",
        "groundtruth": "gt",
    }
    normalized = aliases.get(normalized, normalized)

    if normalized not in VALID_SIMILARITY_MODES:
        raise ValueError(
            f"Unsupported similarity mode: {mode}. "
            f"Expected one of {', '.join(VALID_SIMILARITY_MODES)}"
        )

    return normalized


def compute_cosine_similarity(
    v_pred: torch.Tensor,
    v_ref: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Cosine similarity broadcast back to the original velocity tensor shape."""
    pred_float = v_pred.float()
    ref_float = v_ref.float()

    if pred_float.ndim == 1:
        cosine = F.cosine_similarity(
            pred_float.unsqueeze(0),
            ref_float.unsqueeze(0),
            dim=-1,
            eps=eps,
        )
        similarity = (cosine + 1.0) * 0.5
        return similarity.expand_as(pred_float)

    if pred_float.ndim == 4:
        cosine = F.cosine_similarity(pred_float, ref_float, dim=1, eps=eps)
        similarity = (cosine + 1.0) * 0.5
        return similarity.unsqueeze(1).expand_as(pred_float)

    cosine = F.cosine_similarity(pred_float, ref_float, dim=-1, eps=eps)
    similarity = (cosine + 1.0) * 0.5
    return similarity.unsqueeze(-1).expand_as(pred_float)


def compute_similarity(
    v_pred: torch.Tensor,
    v_ref: torch.Tensor,
    mode: str = "elementwise",
    eps: float = 1e-8,
) -> torch.Tensor:
    similarity_mode = normalize_similarity_mode(mode)

    if similarity_mode == "elementwise":
        return compute_element_similarity(v_pred, v_ref, eps)

    if similarity_mode == "cosine":
        return compute_cosine_similarity(v_pred, v_ref, eps)

    if similarity_mode == "gt":
        raise ValueError("gt mode requires a precomputed gt similarity mask")

    raise ValueError(f"Unsupported similarity mode: {similarity_mode}")


def decode_rle_mask(
    rle_data: Sequence[int],
    height: int = 512,
    width: int = 512,
) -> torch.Tensor:
    """
    Decode RLE mask data from benchmark/mapping_file.json.

    The returned 2D mask uses 1 for edit/foreground region and 0 for
    background/non-edit region.
    """
    mask = torch.zeros(height * width, dtype=torch.float32)
    i = 0
    while i < len(rle_data) - 1:
        start = int(rle_data[i])
        length = int(rle_data[i + 1])
        if start < height * width and length > 0:
            end = min(start + length, height * width)
            mask[start:end] = 1.0
        i += 2
    return mask.reshape(height, width)


def _as_2d_gt_mask(
    gt_mask: Any,
    gt_mask_size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Normalize benchmark GT masks from RLE, tensors, or arrays into a 2D edit mask."""
    height, width = gt_mask_size or (512, 512)

    if isinstance(gt_mask, torch.Tensor):
        mask = gt_mask.detach().float().cpu()
        if mask.ndim == 1 and mask.numel() == height * width:
            mask = mask.reshape(height, width)
        elif mask.ndim == 1:
            mask = decode_rle_mask(mask.tolist(), height, width)
        elif mask.ndim > 2:
            mask = mask.squeeze()
        if mask.ndim != 2:
            raise ValueError(f"GT mask tensor must be 2D after squeeze, got shape {tuple(mask.shape)}")
        return (mask >= 0.5).float()

    if isinstance(gt_mask, (list, tuple)):
        return decode_rle_mask(gt_mask, height, width)

    try:
        mask = torch.as_tensor(gt_mask, dtype=torch.float32)
    except Exception as exc:
        raise TypeError(f"Unsupported GT mask type: {type(gt_mask)}") from exc

    if mask.ndim > 2:
        mask = mask.squeeze()
    if mask.ndim != 2:
        raise ValueError(f"GT mask must be 2D, got shape {tuple(mask.shape)}")
    return (mask >= 0.5).float()


def _infer_token_grid(
    seq_len: int,
    target_height: int,
    target_width: int,
    vae_scale_factor: int,
) -> Tuple[int, int]:
    """Infer the latent token grid used by sequence-shaped model velocities."""
    latent_h = max(1, target_height // vae_scale_factor)
    latent_w = max(1, target_width // vae_scale_factor)
    candidates = [
        (latent_h, latent_w),
        (max(1, latent_h // 2), max(1, latent_w // 2)),
    ]

    for grid_h, grid_w in candidates:
        if grid_h * grid_w == seq_len:
            return grid_h, grid_w

    target_ratio = target_height / max(target_width, 1)
    best = None
    best_error = None
    for grid_h in range(1, int(seq_len ** 0.5) + 1):
        if seq_len % grid_h != 0:
            continue
        for h, w in ((grid_h, seq_len // grid_h), (seq_len // grid_h, grid_h)):
            error = abs((h / max(w, 1)) - target_ratio)
            if best_error is None or error < best_error:
                best = (h, w)
                best_error = error

    if best is not None:
        return best

    raise ValueError(
        "Cannot infer GT mask token grid for "
        f"seq_len={seq_len}, target={target_width}x{target_height}, "
        f"vae_scale_factor={vae_scale_factor}"
    )


def prepare_gt_similarity_mask(
    gt_mask: Any,
    velocity_shape: Tuple[int, ...],
    target_height: int,
    target_width: int,
    vae_scale_factor: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    gt_mask_size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """
    Resize a benchmark GT edit mask to the velocity tensor shape.

    Output values follow similarity semantics:
      background/non-edit = 1.0
      foreground/edit     = 0.0
    """
    edit_mask_2d = _as_2d_gt_mask(gt_mask, gt_mask_size)
    edit_mask = edit_mask_2d.view(1, 1, *edit_mask_2d.shape)
    ndim = len(velocity_shape)

    if ndim == 3:
        batch, seq_len, channels = velocity_shape
        grid_h, grid_w = _infer_token_grid(
            seq_len,
            target_height,
            target_width,
            vae_scale_factor,
        )
        resized = F.interpolate(edit_mask, size=(grid_h, grid_w), mode="nearest")
        # Sequence models store one token per spatial cell; expand the 2D mask
        # across channels so intervention can use the same per-element API.
        edit_flat = (resized.reshape(1, seq_len, 1) >= 0.5).to(torch.float32)
        similarity = 1.0 - edit_flat
        return similarity.expand(batch, seq_len, channels).to(device=device, dtype=dtype)

    if ndim == 4:
        batch, channels, height, width = velocity_shape
        resized = F.interpolate(edit_mask, size=(height, width), mode="nearest")
        # Convolutional latents already expose H/W, so only channel expansion is needed.
        similarity = 1.0 - (resized >= 0.5).to(torch.float32)
        return similarity.expand(batch, channels, height, width).to(device=device, dtype=dtype)

    raise ValueError(f"Unsupported velocity shape for gt mode: {velocity_shape}")


def apply_intervention(
    v_pred: torch.Tensor,
    v_ref: torch.Tensor,
    config: InterventionConfig,
    preserve_active: bool,
    edit_active: bool,
    similarity: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Apply preserve replacement on high-similarity elements and edit blending on low-similarity elements."""
    dtype = v_pred.dtype

    if similarity is None:
        similarity = compute_similarity(
            v_pred,
            v_ref,
            mode=config.similarity_mode,
            eps=eps,
        )

    threshold = config.similarity_threshold
    # High similarity is treated as the preserve region; low similarity is the edit region.
    high_sim_mask = similarity >= threshold
    low_sim_mask = ~high_sim_mask

    v_ref_dtype = v_ref.to(dtype)
    result = v_pred
    preserve_count = 0
    edit_count = 0

    if preserve_active:
        result = torch.where(high_sim_mask, v_ref_dtype, result)
        preserve_count = int(high_sim_mask.sum().item())

    if edit_active:
        a = config.blend_weight
        # a=1 fully trusts the reference velocity, while a=0 keeps the model prediction.
        blended = a * v_ref_dtype + (1 - a) * v_pred
        result = torch.where(low_sim_mask, blended, result)
        edit_count = int(low_sim_mask.sum().item())

    similarity_mask = (similarity < threshold).float()

    return result, similarity_mask, preserve_count, edit_count


def log_intervention_stats(
    step: int,
    sigma: float,
    preserve_active: bool,
    edit_active: bool,
    preserve_count: int,
    edit_count: int,
    total_elements: int,
    similarity_mode: str = "elementwise",
    blend_weight: float = 0.5,
) -> None:
    preserve_ratio = preserve_count / total_elements * 100 if total_elements > 0 else 0
    edit_ratio = edit_count / total_elements * 100 if total_elements > 0 else 0
    preserve_state = "on" if preserve_active else "off"
    edit_state = "on" if edit_active else "off"

    print(
        f"  Step {step}: {similarity_mode} intervention, sigma={sigma:.4f}, "
        f"preserve[{preserve_state}]={preserve_count}/{total_elements} "
        f"({preserve_ratio:.1f}%) replaced, "
        f"edit[{edit_state}]={edit_count}/{total_elements} "
        f"({edit_ratio:.1f}%) blended (a={blend_weight:.2f})"
    )


def log_intervention_summary(
    total_preserve: int,
    total_edit: int,
    preserve_steps: int,
    edit_steps: int,
) -> None:
    total = total_preserve + total_edit
    if total > 0:
        print(
            f"[Intervention Summary] preserve_steps={preserve_steps}, edit_steps={edit_steps}, "
            f"preserve_affected={total_preserve}, edit_affected={total_edit}, "
            f"total_affected={total}"
        )
