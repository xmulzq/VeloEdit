
import torch
from typing import Tuple, Optional
from .types import InterventionConfig


def compute_reference_velocity(
    z_t: torch.Tensor,
    z_0: torch.Tensor,
    sigma: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    return (z_t - z_0) / (sigma + eps)


def compute_element_similarity(
    v_pred: torch.Tensor,
    v_ref: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    pred_float = v_pred.float()
    ref_float = v_ref.float()

    diff_abs = torch.abs(pred_float - ref_float)

    ref_abs = torch.abs(ref_float) + eps

    similarity = ref_abs / (ref_abs + diff_abs)

    return similarity


def apply_intervention(
    v_pred: torch.Tensor,
    v_ref: torch.Tensor,
    config: InterventionConfig,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    dtype = v_pred.dtype

    similarity = compute_element_similarity(v_pred, v_ref, eps)

    threshold = config.similarity_threshold
    high_sim_mask = similarity >= threshold
    low_sim_mask = ~high_sim_mask

    num_high_sim = high_sim_mask.sum().item()

    v_ref_dtype = v_ref.to(dtype)

    result = torch.where(high_sim_mask, v_ref_dtype, v_pred)

    if config.enable_blend:
        a = config.blend_weight
        blended = a * v_ref_dtype + (1 - a) * v_pred
        result = torch.where(low_sim_mask, blended, result)

    similarity_mask = (similarity < threshold).float()

    return result, similarity_mask, int(num_high_sim)


def log_intervention_stats(
    step: int,
    sigma: float,
    num_replaced: int,
    total_elements: int,
    enable_blend: bool = False,
    blend_weight: float = 0.5,
) -> None:
    ratio = num_replaced / total_elements * 100 if total_elements > 0 else 0
    num_low_sim = total_elements - num_replaced

    if enable_blend:
        print(
            f"  Step {step}: sigma={sigma:.4f}, "
            f"high_sim={num_replaced} ({ratio:.1f}%) replaced, "
            f"low_sim={num_low_sim} ({100-ratio:.1f}%) blended (a={blend_weight:.2f})"
        )
    else:
        print(
            f"  Step {step}: Element-wise intervention, sigma={sigma:.4f}, "
            f"replaced {num_replaced}/{total_elements} ({ratio:.1f}%)"
        )


def log_intervention_summary(
    total_replaced: int,
    total_checked: int,
    intervention_steps: int,
) -> None:
    if total_checked > 0:
        ratio = total_replaced / total_checked * 100
        print(
            f"[Intervention Summary] Total replaced: {total_replaced}/{total_checked} "
            f"({ratio:.2f}%) across {intervention_steps} steps"
        )
