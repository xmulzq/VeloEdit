
from typing import Dict, Any, Optional
import torch
import numpy as np
from PIL import Image
from ml_collections import ConfigDict

from diffusers import FluxKontextPipeline
from diffusers.pipelines.flux.pipeline_flux_kontext import (
    retrieve_timesteps,
    calculate_shift,
)

from .base import BaseVelocityAnalyzer
from ..core.sampler import align_first_step_to_reference_step, log_sampling_schedule


class FLUXVelocityAnalyzer(BaseVelocityAnalyzer):
    """FLUX.1-Kontext adapter that exposes transformer velocity predictions."""

    def __init__(
        self,
        config: ConfigDict,
        device: str = "cuda",
        save_tensors: bool = False,
        lora_path: Optional[str] = None,
    ):
        super().__init__(config, device, save_tensors)
        self.lora_path = lora_path

    def load_model(self) -> None:
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }

        config_dtype = self.config.model.dtype
        dtype = dtype_map.get(config_dtype, torch.bfloat16)
        print(f"[FLUX] Using dtype: {dtype}")

        print(f"[FLUX] Loading model from {self.config.model.path}...")
        self.pipeline = FluxKontextPipeline.from_pretrained(
            self.config.model.path,
            torch_dtype=dtype,
        )

        if hasattr(self.pipeline, 'text_encoder') and self.pipeline.text_encoder is not None:
            self.pipeline.text_encoder.to(dtype=dtype)
        if hasattr(self.pipeline, 'text_encoder_2') and self.pipeline.text_encoder_2 is not None:
            self.pipeline.text_encoder_2.to(dtype=dtype)

        self.pipeline.to(self.device)

        if self.lora_path:
            self._load_lora(self.lora_path)

        print("[FLUX] Model loaded successfully.")

    def _load_lora(self, lora_path: str) -> None:
        import os

        if os.path.isfile(lora_path) and lora_path.endswith('.safetensors'):
            print(f"[LoRA] Loading safetensors: {lora_path}")
            self.pipeline.load_lora_weights(lora_path)
        elif os.path.isdir(lora_path):
            print(f"[LoRA] Loading PEFT directory: {lora_path}")
            # PEFT directories wrap the transformer module instead of using diffusers loaders.
            from peft import PeftModel
            self.pipeline.transformer = PeftModel.from_pretrained(
                self.pipeline.transformer, lora_path
            )
        else:
            raise ValueError(f"Invalid LoRA path: {lora_path}")
        print("[LoRA] Loaded successfully.")

    def _prepare_inputs(
        self,
        image: Image.Image,
        prompt: str,
        num_inference_steps: int,
        seed: int,
    ) -> Dict[str, Any]:
        original_height = image.height
        original_width = image.width

        height = original_height
        width = original_width
        max_area = 1024 ** 2

        current_area = height * width

        # if current_area > max_area:
        #     scale = (max_area / current_area) ** 0.5
        #     width = round(width * scale)
        #     height = round(height * scale)
        # Match the Kontext/Qwen benchmark path: run sampling near the 1024^2
        # training resolution, then resize outputs back to the original size.
        scale = (max_area / current_area) ** 0.5
        width = round(width * scale)
        height = round(height * scale)

        multiple_of = self.pipeline.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of

        print(f"[FLUX] Original size: {original_width}x{original_height}, Working size: {width}x{height}")

        device = self.device
        batch_size = 1

        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            prompt_embeds=None,
            pooled_prompt_embeds=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512,
            lora_scale=None,
        )

        resized_input_image = self.pipeline.image_processor.resize(image, height, width)
        processed_image = self.pipeline.image_processor.preprocess(
            resized_input_image, height, width
        )

        num_channels_latents = self.pipeline.transformer.config.in_channels // 4
        processed_image = processed_image.to(device=device, dtype=prompt_embeds.dtype)

        # Match diffusers examples/tests that pass torch.manual_seed(seed), which
        # is a CPU generator and produces different latents from a CUDA generator.
        generator = torch.Generator(device="cpu").manual_seed(seed)
        print(f"[Seed] Using random seed: {seed}")

        with torch.no_grad():
            latents, image_latents, latent_ids, image_ids = self.pipeline.prepare_latents(
                processed_image,
                batch_size,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                None,
            )

        if image_ids is not None:
            # Kontext attends over generated tokens followed by condition-image tokens.
            latent_ids = torch.cat([latent_ids, image_ids], dim=0)

        transformer_dtype = self.pipeline.transformer.dtype
        latents = latents.to(dtype=transformer_dtype)
        if image_latents is not None:
            image_latents = image_latents.to(dtype=transformer_dtype)

        # The condition image latent is the fixed reference used by velocity intervention.
        reference_latent = image_latents.clone() if image_latents is not None else latents.clone()

        requested_num_inference_steps = num_inference_steps
        sigmas = np.linspace(1.0, 1 / requested_num_inference_steps, requested_num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.pipeline.scheduler.config.get("base_image_seq_len", 256),
            self.pipeline.scheduler.config.get("max_image_seq_len", 4096),
            self.pipeline.scheduler.config.get("base_shift", 0.5),
            self.pipeline.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.pipeline.scheduler,
            requested_num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        raw_sigma_schedule = self.pipeline.scheduler.sigmas.float().clone()
        first_step_align_steps = int(
            getattr(self.config.sampling, "first_step_align_steps", 6)
        )
        reference_sigma_schedule = None
        sigma_schedule = raw_sigma_schedule

        if (
            first_step_align_steps > 0
            and requested_num_inference_steps > first_step_align_steps
        ):
            reference_sigmas = np.linspace(
                1.0,
                1 / first_step_align_steps,
                first_step_align_steps,
            )
            retrieve_timesteps(
                self.pipeline.scheduler,
                first_step_align_steps,
                device,
                sigmas=reference_sigmas,
                mu=mu,
            )
            reference_sigma_schedule = self.pipeline.scheduler.sigmas.float().clone()
            sigma_schedule = align_first_step_to_reference_step(
                raw_sigma_schedule,
                reference_sigma_schedule,
                first_step_align_steps,
            )

        # Log both requested and effective schedules because first-step alignment can
        # change the actual number of Euler transitions.
        log_sampling_schedule(
            "FLUX",
            requested_num_inference_steps,
            raw_sigma_schedule,
            sigma_schedule,
            reference_sigma_schedule,
            first_step_align_steps,
        )

        guidance_scale = self.config.sampling.guidance_scale
        if self.pipeline.transformer.config.guidance_embeds:
            guidance = torch.full(
                [1], guidance_scale, device=device, dtype=torch.float32
            ).expand(latents.shape[0])
        else:
            guidance = None

        def v_pred_fn(z, sigma):
            # The sampler owns z, while the condition image latent stays fixed per prompt.
            latent_model_input = z
            if image_latents is not None:
                latent_model_input = torch.cat([z, image_latents], dim=1)

            train_timesteps = self.pipeline.scheduler.config.get("num_train_timesteps", 1000)
            sigma_tensor = torch.as_tensor(sigma, device=z.device, dtype=torch.float32)
            # FLUX transformer expects normalized training timesteps, not raw sigma values.
            timesteps_input = (
                (sigma_tensor * train_timesteps)
                .expand(latent_model_input.shape[0])
                .to(dtype=z.dtype)
                / train_timesteps
            )
            noise_pred = self.pipeline.transformer(
                hidden_states=latent_model_input,
                timestep=timesteps_input,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_ids,
                joint_attention_kwargs={},
                return_dict=False,
            )[0]
            # Drop condition-image predictions; intervention only edits generated tokens.
            noise_pred = noise_pred[:, : latents.size(1)]
            return noise_pred

        return {
            'latents': latents,
            'reference_latent': reference_latent,
            'sigma_schedule': sigma_schedule,
            'v_pred_fn': v_pred_fn,
            'height': height,
            'width': width,
            'original_height': original_height,
            'original_width': original_width,
            'input_image_resized': resized_input_image,
        }

    def _decode_latents(
        self,
        latents: torch.Tensor,
        height: int,
        width: int,
    ) -> Image.Image:
        latents = latents.to(device=self.device, dtype=self.pipeline.transformer.dtype)

        # FLUX stores generated latents as packed tokens before VAE unpacking.
        unpacked = self.pipeline._unpack_latents(
            latents, height, width, self.pipeline.vae_scale_factor
        )

        for_vae = (
            unpacked / self.pipeline.vae.config.scaling_factor
        ) + self.pipeline.vae.config.shift_factor
        for_vae = for_vae.to(dtype=self.pipeline.vae.dtype)

        with torch.no_grad():
            decoded = self.pipeline.vae.decode(for_vae, return_dict=False)[0]

        image = self.pipeline.image_processor.postprocess(
            decoded, output_type="pil"
        )[0]

        return image
