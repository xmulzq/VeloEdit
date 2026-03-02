
import torch
from typing import Callable, Optional, List, Tuple
from functools import partial
import torch.distributed as dist
import tqdm

from .types import SamplingResult, InterventionConfig
from .intervention import (
    compute_reference_velocity,
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
    dt = sigma_next - sigma

    x0_pred = z - sigma * v_pred

    z_next = z + dt * v_pred

    return z_next, x0_pred


def run_deterministic_sampling(
    v_pred_fn: Callable[[torch.Tensor, float], torch.Tensor],
    z: torch.Tensor,
    sigma_schedule: torch.Tensor,
    reference_latent: Optional[torch.Tensor] = None,
    intervention_config: Optional[InterventionConfig] = None,
) -> SamplingResult:
    dtype = z.dtype
    device = z.device

    all_latents = [z.detach().clone()]
    all_velocities = []
    step_pred_x0 = []
    similarity_masks = []
    sigmas = [sigma_schedule[i].item() for i in range(len(sigma_schedule))]

    if intervention_config is None:
        intervention_config = InterventionConfig()
    enable_intervention = intervention_config.is_enabled() and reference_latent is not None
    generate_mask = reference_latent is not None

    total_replaced = 0
    total_checked = 0

    num_steps = len(sigma_schedule) - 1
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

        if generate_mask:
            v_ref = compute_reference_velocity(z, reference_latent, sigma_val)
            v_ref = v_ref.to(dtype)

            from .intervention import compute_element_similarity
            similarity = compute_element_similarity(v_pred, v_ref)
            step_mask = (similarity < intervention_config.similarity_threshold).float()
            similarity_masks.append(step_mask.detach().clone().cpu())

        if enable_intervention and i < intervention_config.steps:
            v_ref = compute_reference_velocity(z, reference_latent, sigma_val)
            v_ref = v_ref.to(dtype)

            v_pred, _, num_replaced = apply_intervention(
                v_pred, v_ref, intervention_config
            )

            num_elements = v_pred.numel()
            total_replaced += num_replaced
            total_checked += num_elements

            log_intervention_stats(
                step=i,
                sigma=sigma_val,
                num_replaced=num_replaced,
                total_elements=num_elements,
                enable_blend=intervention_config.enable_blend,
                blend_weight=intervention_config.blend_weight,
            )

        all_velocities.append(v_pred.detach().clone().cpu())

        z, x0_pred = euler_step(z, v_pred, sigma_val, sigma_next_val)
        z = z.to(dtype)

        step_pred_x0.append(x0_pred.detach().clone().cpu())
        all_latents.append(z.detach().clone().cpu())

    if total_checked > 0:
        log_intervention_summary(total_replaced, total_checked, intervention_config.steps)

    return SamplingResult(
        latents=z.to(dtype),
        all_latents=all_latents,
        all_velocities=all_velocities,
        step_pred_x0=step_pred_x0,
        sigmas=sigmas,
        similarity_masks=similarity_masks if generate_mask else None,
        interventions_applied=total_replaced,
    )


def create_sigma_schedule(
    num_steps: int,
    sigma_max: float = 1.0,
    sigma_min: float = 0.0,
) -> torch.Tensor:
    return torch.linspace(sigma_max, sigma_min, num_steps + 1)
