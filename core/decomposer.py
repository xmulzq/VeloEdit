
import math
import torch
from typing import List, Dict, Any

from .types import VelocityDecomposition


class VelocityDecomposer:

    def __init__(self, eps: float = 1e-8, save_tensors: bool = False):
        self.eps = eps
        self.save_tensors = save_tensors

    def decompose(
        self,
        velocity: torch.Tensor,
        current_latent: torch.Tensor,
        reference_velocity: torch.Tensor,
        step: int,
        sigma: float,
    ) -> VelocityDecomposition:
        device = velocity.device
        current_latent = current_latent.to(device)
        reference_velocity = reference_velocity.to(device)

        v_flat = velocity.flatten(1)
        v_ref_flat = reference_velocity.flatten(1)

        v_ref_norm = torch.norm(v_ref_flat, dim=1, keepdim=True)
        v_ref_normalized = v_ref_flat / (v_ref_norm + self.eps)

        proj_coef = (v_flat * v_ref_normalized).sum(dim=1, keepdim=True)

        preserve_flat = proj_coef * v_ref_normalized
        preserve_component = preserve_flat.view_as(velocity)

        edit_flat = v_flat - preserve_flat
        edit_component = edit_flat.view_as(velocity)

        total_magnitude = torch.norm(v_flat, dim=1).mean().item()
        preserve_magnitude = torch.norm(preserve_flat, dim=1).mean().item()
        edit_magnitude = torch.norm(edit_flat, dim=1).mean().item()

        total_energy = (v_flat ** 2).sum(dim=1).mean().item()
        preserve_energy = (preserve_flat ** 2).sum(dim=1).mean().item()
        edit_energy = (edit_flat ** 2).sum(dim=1).mean().item()

        preserve_ratio = preserve_energy / (total_energy + self.eps)
        edit_ratio = edit_energy / (total_energy + self.eps)

        v_norm = torch.norm(v_flat, dim=1, keepdim=True)
        cos_angle = (v_flat * v_ref_flat).sum(dim=1, keepdim=True) / (v_norm * v_ref_norm + self.eps)
        cos_angle = cos_angle.mean().clamp(-1, 1)
        angle_rad = torch.acos(cos_angle)
        angle_deg = angle_rad.item() * 180 / math.pi

        result = VelocityDecomposition(
            step=step,
            sigma=sigma,
            preserve_magnitude=preserve_magnitude,
            edit_magnitude=edit_magnitude,
            total_magnitude=total_magnitude,
            preserve_ratio=preserve_ratio,
            edit_ratio=edit_ratio,
            angle_to_reference=angle_deg,
            distance_to_ref=0.0,
        )

        if self.save_tensors:
            result.velocity = velocity.detach().cpu()
            result.preserve_component = preserve_component.detach().cpu()
            result.edit_component = edit_component.detach().cpu()

        return result

    def decompose_trajectory(
        self,
        velocities: List[torch.Tensor],
        latents: List[torch.Tensor],
        reference_latent: torch.Tensor,
        sigmas: List[float],
    ) -> List[VelocityDecomposition]:
        results = []

        for i, velocity in enumerate(velocities):
            current_latent = latents[i]
            sigma = sigmas[i]

            device = velocity.device
            current_latent = current_latent.to(device)
            ref_latent = reference_latent.to(device)

            reference_velocity = (current_latent - ref_latent) / (sigma + self.eps)

            result = self.decompose(
                velocity=velocity,
                current_latent=current_latent,
                reference_velocity=reference_velocity,
                step=i,
                sigma=sigma,
            )

            current_flat = current_latent.flatten(1)
            ref_flat = ref_latent.flatten(1)
            result.distance_to_ref = torch.norm(ref_flat - current_flat, dim=1).mean().item()

            results.append(result)

        return results

    @staticmethod
    def compute_summary(decompositions: List[VelocityDecomposition]) -> Dict[str, Any]:
        if not decompositions:
            return {}

        preserve_ratios = [d.preserve_ratio for d in decompositions]
        edit_ratios = [d.edit_ratio for d in decompositions]
        angles = [d.angle_to_reference for d in decompositions]
        distances = [d.distance_to_ref for d in decompositions]

        n = len(decompositions)
        half = n // 2

        return {
            "mean_preserve_ratio": sum(preserve_ratios) / n,
            "mean_edit_ratio": sum(edit_ratios) / n,
            "mean_angle": sum(angles) / n,
            "mean_distance": sum(distances) / n,

            "preserve_ratio_trajectory": preserve_ratios,
            "edit_ratio_trajectory": edit_ratios,
            "angle_trajectory": angles,
            "distance_trajectory": distances,

            "early_preserve_ratio": sum(preserve_ratios[:half]) / max(1, half),
            "late_preserve_ratio": sum(preserve_ratios[half:]) / max(1, n - half),
            "early_edit_ratio": sum(edit_ratios[:half]) / max(1, half),
            "late_edit_ratio": sum(edit_ratios[half:]) / max(1, n - half),

            "preserve_ratio_delta": preserve_ratios[-1] - preserve_ratios[0] if n > 1 else 0,
            "edit_ratio_delta": edit_ratios[-1] - edit_ratios[0] if n > 1 else 0,
        }
