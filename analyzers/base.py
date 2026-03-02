
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import torch
import numpy as np
from PIL import Image
from ml_collections import ConfigDict

from ..core.types import (
    InterventionConfig,
    SamplingResult,
    VelocityDecomposition,
    AnalysisResult,
)
from ..core.sampler import run_deterministic_sampling
from ..core.decomposer import VelocityDecomposer


class BaseVelocityAnalyzer(ABC):

    def __init__(
        self,
        config: ConfigDict,
        device: str = "cuda",
        save_tensors: bool = False,
    ):
        self.config = config
        self.device = device
        self.decomposer = VelocityDecomposer(save_tensors=save_tensors)
        self.pipeline = None
        self.model_loaded = False

    @abstractmethod
    def load_model(self) -> None:
        pass

    @abstractmethod
    def _prepare_inputs(
        self,
        image: Image.Image,
        prompt: str,
        num_inference_steps: int,
        seed: int,
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _decode_latents(
        self,
        latents: torch.Tensor,
        height: int,
        width: int,
    ) -> Image.Image:
        pass

    def _parse_intervention_config(
        self,
        intervention_config: Optional[Dict],
    ) -> InterventionConfig:
        if intervention_config is None:
            return InterventionConfig()

        return InterventionConfig(
            steps=intervention_config.get("intervention_steps", 0),
            similarity_threshold=intervention_config.get("similarity_threshold", 0.8),
            enable_blend=intervention_config.get("enable_blend", False),
            blend_weight=intervention_config.get("blend_weight", 0.5),
        )

    def _convert_masks_to_images(
        self,
        masks: Optional[List[torch.Tensor]],
        target_height: int,
        target_width: int,
        vae_scale_factor: int,
    ) -> List[Image.Image]:
        if masks is None or len(masks) == 0:
            return []

        mask_images = []
        latent_h = target_height // vae_scale_factor
        latent_w = target_width // vae_scale_factor

        for mask in masks:
            mask_np = mask[0].numpy()

            if len(mask_np.shape) > 1:
                mask_avg = mask_np.mean(axis=-1)
            else:
                mask_avg = mask_np

            try:
                mask_2d = mask_avg.reshape(latent_h, latent_w)
            except ValueError:
                packed_h = latent_h // 2
                packed_w = latent_w // 2
                if len(mask_avg) == packed_h * packed_w:
                    mask_2d = mask_avg.reshape(packed_h, packed_w)
                else:
                    print(f"[Mask] Warning: Cannot reshape mask of size {len(mask_avg)}")
                    continue

            mask_pil = Image.fromarray((mask_2d * 255).astype(np.uint8), mode='L')
            mask_resized = mask_pil.resize(
                (target_width, target_height),
                Image.Resampling.NEAREST
            )
            mask_images.append(mask_resized)

        return mask_images

    def _convert_masks_to_heatmaps(
        self,
        masks: Optional[List[torch.Tensor]],
        target_height: int,
        target_width: int,
        vae_scale_factor: int,
        colormap: str = "viridis",
    ) -> List[Image.Image]:
        if masks is None or len(masks) == 0:
            return []

        try:
            import matplotlib.cm as cm
        except ImportError:
            print("[Heatmap] Warning: matplotlib not available")
            return []

        heatmap_images = []
        latent_h = target_height // vae_scale_factor
        latent_w = target_width // vae_scale_factor
        cmap = cm.get_cmap(colormap)

        for mask in masks:
            mask_np = mask[0].numpy()

            if len(mask_np.shape) > 1:
                mask_avg = mask_np.mean(axis=-1)
            else:
                mask_avg = mask_np

            try:
                mask_2d = mask_avg.reshape(latent_h, latent_w)
            except ValueError:
                packed_h = latent_h // 2
                packed_w = latent_w // 2
                if len(mask_avg) == packed_h * packed_w:
                    mask_2d = mask_avg.reshape(packed_h, packed_w)
                else:
                    continue

            colored = cmap(mask_2d)
            colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)

            heatmap_pil = Image.fromarray(colored_rgb, mode='RGB')
            heatmap_resized = heatmap_pil.resize(
                (target_width, target_height),
                Image.Resampling.BILINEAR
            )
            heatmap_images.append(heatmap_resized)

        return heatmap_images

    def analyze(
        self,
        image: Image.Image,
        prompt: str,
        image_path: str = "",
        intervention_config: Optional[Dict] = None,
    ) -> AnalysisResult:
        if not self.model_loaded:
            self.load_model()
            self.model_loaded = True

        num_inference_steps = self.config.sampling.num_inference_steps
        if intervention_config and "num_inference_steps" in intervention_config:
            num_inference_steps = intervention_config["num_inference_steps"]

        seed = 42
        if intervention_config and "seed" in intervention_config:
            seed = intervention_config["seed"]

        config = self._parse_intervention_config(intervention_config)

        inputs = self._prepare_inputs(image, prompt, num_inference_steps, seed)

        with torch.no_grad():
            sampling_result = run_deterministic_sampling(
                v_pred_fn=inputs['v_pred_fn'],
                z=inputs['latents'],
                sigma_schedule=inputs['sigma_schedule'],
                reference_latent=inputs['reference_latent'],
                intervention_config=config,
            )

        generated_image = self._decode_latents(
            sampling_result.latents,
            inputs['height'],
            inputs['width'],
        )

        original_width = inputs.get('original_width', inputs['width'])
        original_height = inputs.get('original_height', inputs['height'])
        need_resize = (original_width != inputs['width'] or original_height != inputs['height'])

        if need_resize:
            generated_image = generated_image.resize(
                (original_width, original_height),
                Image.Resampling.LANCZOS
            )
            print(f"[Resize] Output resized to original: {original_width}x{original_height}")

        print(f"[Step Images] Decoding {len(sampling_result.step_pred_x0)} step predictions...")
        step_images = []
        for x0_pred in sampling_result.step_pred_x0:
            step_img = self._decode_latents(
                x0_pred.to(self.device),
                inputs['height'],
                inputs['width'],
            )
            if need_resize:
                step_img = step_img.resize(
                    (original_width, original_height),
                    Image.Resampling.LANCZOS
                )
            step_images.append(step_img)
        print(f"[Step Images] Successfully decoded {len(step_images)} images")

        decompositions = self.decomposer.decompose_trajectory(
            velocities=sampling_result.all_velocities,
            latents=sampling_result.all_latents,
            reference_latent=inputs['reference_latent'],
            sigmas=sampling_result.sigmas,
        )

        vae_scale_factor = getattr(self.pipeline, 'vae_scale_factor', 8)
        mask_images = self._convert_masks_to_images(
            sampling_result.similarity_masks,
            inputs['height'],
            inputs['width'],
            vae_scale_factor,
        )
        heatmap_images = self._convert_masks_to_heatmaps(
            sampling_result.similarity_masks,
            inputs['height'],
            inputs['width'],
            vae_scale_factor,
        )

        if need_resize and mask_images:
            mask_images = [
                img.resize((original_width, original_height), Image.Resampling.NEAREST)
                for img in mask_images
            ]
        if need_resize and heatmap_images:
            heatmap_images = [
                img.resize((original_width, original_height), Image.Resampling.BILINEAR)
                for img in heatmap_images
            ]

        if mask_images:
            print(f"[Similarity Masks] Generated {len(mask_images)} mask images")
        if heatmap_images:
            print(f"[Similarity Heatmaps] Generated {len(heatmap_images)} heatmap images")

        summary = VelocityDecomposer.compute_summary(decompositions)

        return AnalysisResult(
            image_path=image_path,
            prompt=prompt,
            model_name=self.config.model.name,
            num_steps=len(decompositions),
            sigmas=sampling_result.sigmas,
            decompositions=decompositions,
            summary=summary,
            generated_image=generated_image,
            step_images=step_images,
            similarity_mask_images=mask_images if mask_images else None,
            similarity_heatmap_images=heatmap_images if heatmap_images else None,
            interventions_applied=sampling_result.interventions_applied,
        )

    def analyze_batch(
        self,
        images: List[Image.Image],
        prompts: List[str],
        image_paths: Optional[List[str]] = None,
        intervention_config: Optional[Dict] = None,
    ) -> List[AnalysisResult]:
        if image_paths is None:
            image_paths = [""] * len(images)

        results = []
        for i, (img, prompt, path) in enumerate(zip(images, prompts, image_paths)):
            print(f"Analyzing {i+1}/{len(images)}: {path or 'image'}")
            result = self.analyze(img, prompt, path, intervention_config)
            results.append(result)

        return results
