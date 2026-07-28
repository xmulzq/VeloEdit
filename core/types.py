
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import torch


@dataclass
class InterventionConfig:
    """Runtime controls for velocity-field intervention."""

    # Step counts are resolved against the effective sigma schedule; negative values
    # are allowed by the sampler and mean "effective_steps + value".
    preserve_steps: int = 0
    edit_steps: int = 0
    similarity_threshold: float = 0.8
    similarity_mode: str = "elementwise"
    enable_interv: bool = True
    # Edit-region velocity is blended as a * v_ref + (1 - a) * v_pred.
    blend_weight: float = 0.5
    # Optional benchmark mask fields, used only when similarity_mode == "gt".
    gt_mask: Optional[Any] = None
    gt_mask_size: Optional[Tuple[int, int]] = None
    gt_target_height: Optional[int] = None
    gt_target_width: Optional[int] = None
    gt_vae_scale_factor: int = 8

    def is_enabled(self) -> bool:
        return self.enable_interv and (self.preserve_steps != 0 or self.edit_steps != 0)


@dataclass
class SamplingResult:
    """Full sampler trace used by analysis, visualization, and benchmarking."""

    latents: torch.Tensor
    all_latents: List[torch.Tensor]
    all_velocities: List[torch.Tensor]
    step_pred_x0: List[torch.Tensor]
    sigmas: List[float]
    similarity_masks: Optional[List[torch.Tensor]] = None
    interventions_applied: int = 0
    preserve_interventions_applied: int = 0
    edit_interventions_applied: int = 0
    velocity_intervention_time_sec: float = 0.0


@dataclass
class VelocityDecomposition:
    """Per-step projection metrics for a predicted velocity."""

    step: int
    sigma: float

    preserve_magnitude: float
    edit_magnitude: float
    total_magnitude: float

    preserve_ratio: float
    edit_ratio: float

    angle_to_reference: float
    distance_to_ref: float

    velocity: Optional[torch.Tensor] = None
    preserve_component: Optional[torch.Tensor] = None
    edit_component: Optional[torch.Tensor] = None


@dataclass
class AnalysisResult:
    """Analyzer output shared by CLI, server API, and benchmark scripts."""

    image_path: str
    prompt: str
    model_name: str
    num_steps: int
    sigmas: List[float]
    decompositions: List[VelocityDecomposition]
    summary: Dict[str, Any]

    generated_image: Any = None
    step_images: Optional[List[Any]] = None
    similarity_mask_images: Optional[List[Any]] = None
    similarity_heatmap_images: Optional[List[Any]] = None

    interventions_applied: int = 0
    preserve_interventions_applied: int = 0
    edit_interventions_applied: int = 0
    inference_time_sec: float = 0.0
    velocity_intervention_time_sec: float = 0.0
    velocity_intervention_time_ratio: float = 0.0
