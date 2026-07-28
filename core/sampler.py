
import torch
from typing import Callable, Optional, List, Tuple
from functools import partial
import torch.distributed as dist
import tqdm
import time

from .types import SamplingResult, InterventionConfig
from .intervention import (
    compute_reference_velocity,
    compute_similarity,
    normalize_similarity_mode,
    prepare_gt_similarity_mask,
    apply_intervention,
    log_intervention_stats,
    log_intervention_summary,
)

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


def euler_step(
    z: torch.Tensor,
    v_pred: torch.Tensor,
    sigma: float,
    sigma_next: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One deterministic Euler update for flow-matching style velocity prediction."""
    sample = z.to(torch.float32)
    model_output = v_pred.to(torch.float32)
    current_sigma = torch.as_tensor(sigma, device=z.device, dtype=torch.float32)
    next_sigma = torch.as_tensor(sigma_next, device=z.device, dtype=torch.float32)
    dt = next_sigma - current_sigma

    x0_pred = sample - current_sigma * model_output

    z_next = sample + dt * model_output

    return z_next.to(v_pred.dtype), x0_pred


def _sigma_values(sigmas: torch.Tensor) -> List[float]:
    return [
        sigma.item() if hasattr(sigma, "item") else float(sigma)
        for sigma in sigmas
    ]


def compute_sigma_deltas(sigmas: torch.Tensor) -> List[float]:
    values = _sigma_values(sigmas)
    return [values[i] - values[i + 1] for i in range(len(values) - 1)]


def format_sigma_schedule(sigmas: torch.Tensor) -> str:
    return "[" + ", ".join(f"{value:.6f}" for value in _sigma_values(sigmas)) + "]"


def format_sigma_deltas(sigmas: torch.Tensor) -> str:
    return "[" + ", ".join(f"{value:.6f}" for value in compute_sigma_deltas(sigmas)) + "]"


def _should_log() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _synchronize_if_cuda(device: torch.device) -> None:
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def align_first_step_to_reference_step(
    sigma_schedule_n: torch.Tensor,
    sigma_schedule_reference: torch.Tensor,
    reference_steps: int,
    atol: float = 1e-6,
) -> torch.Tensor:
    """Force the first denoising delta to match a lower-step reference schedule."""
    requested_steps = len(sigma_schedule_n) - 1
    if (
        reference_steps <= 0
        or requested_steps <= reference_steps
        or len(sigma_schedule_n) < 2
        or len(sigma_schedule_reference) < 2
    ):
        return sigma_schedule_n

    sigma_n_float = sigma_schedule_n.detach().float()
    sigma_reference_float = sigma_schedule_reference.detach().float()
    reference_delta = sigma_reference_float[0] - sigma_reference_float[1]
    target_sigma1_float = sigma_n_float[0] - reference_delta

    start = sigma_n_float[0].item()
    end = sigma_n_float[-1].item()
    target = target_sigma1_float.item()
    descending = start >= end

    if descending:
        if target >= start or target <= end:
            return sigma_schedule_n
        tail_mask = sigma_n_float < (target - atol)
    else:
        if target <= start or target >= end:
            return sigma_schedule_n
        tail_mask = sigma_n_float > (target + atol)

    target_sigma1 = target_sigma1_float.to(
        device=sigma_schedule_n.device,
        dtype=sigma_schedule_n.dtype,
    ).reshape(1)
    # Keep the original start/end while replacing only the first transition.
    effective = torch.cat(
        [sigma_schedule_n[:1], target_sigma1, sigma_schedule_n[tail_mask]],
        dim=0,
    )

    return effective


def log_sampling_schedule(
    model_name: str,
    requested_steps: int,
    raw_sigma_schedule: torch.Tensor,
    effective_sigma_schedule: torch.Tensor,
    reference_sigma_schedule: Optional[torch.Tensor] = None,
    first_step_align_steps: Optional[int] = None,
) -> None:
    if not _should_log():
        return

    print(f"[Sigma Schedule][{model_name}] requested_steps={requested_steps}")
    if reference_sigma_schedule is not None:
        reference_label = (
            first_step_align_steps
            if first_step_align_steps is not None
            else len(reference_sigma_schedule) - 1
        )
        print(
            f"[Sigma Schedule][{model_name}] {reference_label}-step reference sigmas: "
            f"{format_sigma_schedule(reference_sigma_schedule)}"
        )
        print(
            f"[Sigma Schedule][{model_name}] {reference_label}-step reference deltas: "
            f"{format_sigma_deltas(reference_sigma_schedule)}"
        )
    print(
        f"[Sigma Schedule][{model_name}] requested raw sigmas: "
        f"{format_sigma_schedule(raw_sigma_schedule)}"
    )
    print(
        f"[Sigma Schedule][{model_name}] requested raw deltas: "
        f"{format_sigma_deltas(raw_sigma_schedule)}"
    )
    print(
        f"[Sigma Schedule][{model_name}] effective_steps={len(effective_sigma_schedule) - 1}"
    )
    print(
        f"[Sigma Schedule][{model_name}] effective sigmas: "
        f"{format_sigma_schedule(effective_sigma_schedule)}"
    )
    print(
        f"[Sigma Schedule][{model_name}] effective deltas: "
        f"{format_sigma_deltas(effective_sigma_schedule)}"
    )


def resolve_intervention_steps(requested_steps: int, num_steps: int) -> int:
    if requested_steps < 0:
        requested_steps = num_steps + requested_steps
    return max(0, min(requested_steps, num_steps))


def run_deterministic_sampling(
    v_pred_fn: Callable[[torch.Tensor, float], torch.Tensor],
    z: torch.Tensor,
    sigma_schedule: torch.Tensor,
    reference_latent: Optional[torch.Tensor] = None,
    intervention_config: Optional[InterventionConfig] = None,
) -> SamplingResult:
    """Run deterministic sampling and optionally intervene in the predicted velocity field."""
    dtype = z.dtype
    device = z.device

    all_latents = [z.detach().clone()]
    all_velocities = []
    step_pred_x0 = []
    similarity_masks = []
    sigmas = [sigma_schedule[i].item() for i in range(len(sigma_schedule))]

    if intervention_config is None:
        intervention_config = InterventionConfig()
    similarity_mode = normalize_similarity_mode(intervention_config.similarity_mode)
    # Intervention needs a fixed reference latent; without it we still sample normally.
    enable_intervention = intervention_config.is_enabled() and reference_latent is not None
    generate_mask = enable_intervention
    gt_similarity = None

    total_preserve = 0
    total_edit = 0
    velocity_intervention_time_sec = 0.0

    num_steps = len(sigma_schedule) - 1
    preserve_steps = resolve_intervention_steps(
        intervention_config.preserve_steps,
        num_steps,
    )
    edit_steps = resolve_intervention_steps(
        intervention_config.edit_steps,
        num_steps,
    )
    if _should_log() and (
        preserve_steps != intervention_config.preserve_steps
        or edit_steps != intervention_config.edit_steps
    ):
        print(
            "[Intervention Steps] "
            f"effective_sampling_steps={num_steps}, "
            f"preserve={intervention_config.preserve_steps}->{preserve_steps}, "
            f"edit={intervention_config.edit_steps}->{edit_steps}"
        )

    for i in tqdm(
        range(num_steps),
        desc="Deterministic Sampling",
        disable=dist.is_initialized() and dist.get_rank() != 0,
    ):
        sigma = sigma_schedule[i]
        sigma_next = sigma_schedule[i + 1]
        sigma_val = sigma.item() if hasattr(sigma, 'item') else float(sigma)
        sigma_next_val = sigma_next.item() if hasattr(sigma_next, 'item') else float(sigma_next)

        v_pred = v_pred_fn(z.to(dtype), sigma)
        v_ref = None
        similarity = None

        if generate_mask:
            _synchronize_if_cuda(v_pred.device)
            intervention_start = time.perf_counter()
            # The reference velocity is recomputed from the current latent so the mask
            # always describes the current sampling state, not just the initial noise.
            v_ref = compute_reference_velocity(z, reference_latent, sigma_val)
            v_ref = v_ref.to(dtype)

            if similarity_mode == "gt":
                if intervention_config.gt_mask is None:
                    raise ValueError(
                        "gt similarity mode requires gt_mask from benchmark mapping_file.json"
                    )
                if gt_similarity is None:
                    if (
                        intervention_config.gt_target_height is None
                        or intervention_config.gt_target_width is None
                    ):
                        raise ValueError(
                            "gt similarity mode requires target height/width from analyzer inputs"
                        )
                    gt_similarity = prepare_gt_similarity_mask(
                        gt_mask=intervention_config.gt_mask,
                        velocity_shape=tuple(v_pred.shape),
                        target_height=intervention_config.gt_target_height,
                        target_width=intervention_config.gt_target_width,
                        vae_scale_factor=intervention_config.gt_vae_scale_factor,
                        device=v_pred.device,
                        dtype=torch.float32,
                        gt_mask_size=intervention_config.gt_mask_size,
                    )
                # GT masks are spatial and step-invariant, so reuse the resized tensor.
                similarity = gt_similarity
            else:
                similarity = compute_similarity(
                    v_pred,
                    v_ref,
                    mode=similarity_mode,
                )
            step_mask = (similarity < intervention_config.similarity_threshold).float()
            _synchronize_if_cuda(v_pred.device)
            velocity_intervention_time_sec += time.perf_counter() - intervention_start
            similarity_masks.append(step_mask.detach().clone().cpu())

        preserve_active = enable_intervention and i < preserve_steps
        edit_active = enable_intervention and i < edit_steps

        if (preserve_active or edit_active) and v_ref is not None:
            _synchronize_if_cuda(v_pred.device)
            intervention_start = time.perf_counter()
            v_pred, _, preserve_count, edit_count = apply_intervention(
                v_pred,
                v_ref,
                intervention_config,
                preserve_active=preserve_active,
                edit_active=edit_active,
                similarity=similarity,
            )
            _synchronize_if_cuda(v_pred.device)
            velocity_intervention_time_sec += time.perf_counter() - intervention_start

            num_elements = v_pred.numel()
            total_preserve += preserve_count
            total_edit += edit_count

            log_intervention_stats(
                step=i,
                sigma=sigma_val,
                preserve_active=preserve_active,
                edit_active=edit_active,
                preserve_count=preserve_count,
                edit_count=edit_count,
                total_elements=num_elements,
                similarity_mode=similarity_mode,
                blend_weight=intervention_config.blend_weight,
            )

        # Store CPU copies for downstream decomposition/visualization without
        # keeping the full trajectory resident on GPU.
        all_velocities.append(v_pred.detach().clone().cpu())

        z, x0_pred = euler_step(z, v_pred, sigma_val, sigma_next_val)
        z = z.to(dtype)

        step_pred_x0.append(x0_pred.detach().clone().cpu())
        all_latents.append(z.detach().clone().cpu())

    if total_preserve + total_edit > 0:
        log_intervention_summary(
            total_preserve,
            total_edit,
            preserve_steps,
            edit_steps,
        )

    return SamplingResult(
        latents=z.to(dtype),
        all_latents=all_latents,
        all_velocities=all_velocities,
        step_pred_x0=step_pred_x0,
        sigmas=sigmas,
        similarity_masks=similarity_masks if generate_mask else None,
        interventions_applied=total_preserve + total_edit,
        preserve_interventions_applied=total_preserve,
        edit_interventions_applied=total_edit,
        velocity_intervention_time_sec=velocity_intervention_time_sec,
    )


def create_sigma_schedule(
    num_steps: int,
    sigma_max: float = 1.0,
    sigma_min: float = 0.0,
) -> torch.Tensor:
    return torch.linspace(sigma_max, sigma_min, num_steps + 1)
