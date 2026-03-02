
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import torch


@dataclass
class InterventionConfig:
    steps: int = 0
    similarity_threshold: float = 0.8
    enable_blend: bool = False
    blend_weight: float = 0.5

    def is_enabled(self) -> bool:
        return self.steps > 0


@dataclass
class SamplingResult:
    latents: torch.Tensor
    all_latents: List[torch.Tensor]
    all_velocities: List[torch.Tensor]
    step_pred_x0: List[torch.Tensor]
    sigmas: List[float]
    similarity_masks: Optional[List[torch.Tensor]] = None
    interventions_applied: int = 0


@dataclass
class VelocityDecomposition:
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
