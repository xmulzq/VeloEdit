
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


class FLUXVelocityAnalyzer(BaseVelocityAnalyzer):

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

        self.pipeline.vae.to(dtype=torch.float32)
        print("[FLUX] Model loaded successfully.")

    def _load_lora(self, lora_path: str) -> None:
        import os

        if os.path.isfile(lora_path) and lora_path.endswith('.safetensors'):
            print(f"[LoRA] Loading safetensors: {lora_path}")
            self.pipeline.load_lora_weights(lora_path)
        elif os.path.isdir(lora_path):
            print(f"[LoRA] Loading PEFT directory: {lora_path}")
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
        if current_area > max_area:
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
        processed_image = processed_image.to(device=device, dtype=torch.float32)

        generator = torch.Generator(device=device).manual_seed(seed)
        print(f"[Seed] Using random seed: {seed}")

        with torch.no_grad():
            latents, image_latents, latent_ids, image_ids = self.pipeline.prepare_latents(
                processed_image,
                batch_size,
                num_channels_latents,
                height,
                width,
                torch.float32,
                device,
                generator,
                None,
            )

        if image_ids is not None:
            latent_ids = torch.cat([latent_ids, image_ids], dim=0)

        transformer_dtype = self.pipeline.transformer.dtype
        latents = latents.to(dtype=transformer_dtype)
        if image_latents is not None:
            image_latents = image_latents.to(dtype=transformer_dtype)

        reference_latent = image_latents.clone() if image_latents is not None else latents.clone()

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
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
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        sigma_schedule = self.pipeline.scheduler.sigmas.float()

        guidance_scale = self.config.sampling.guidance_scale
        if self.pipeline.transformer.config.guidance_embeds:
            guidance = torch.full(
                [1], guidance_scale, device=device, dtype=torch.float32
            ).expand(latents.shape[0])
        else:
            guidance = None

        def v_pred_fn(z, sigma):
            latent_model_input = z
            if image_latents is not None:
                latent_model_input = torch.cat([z, image_latents], dim=1)

            timesteps_input = torch.full(
                [latent_model_input.shape[0]], sigma, device=z.device, dtype=torch.float32
            )
            noise_pred = self.pipeline.transformer(
                hidden_states=latent_model_input,
                timestep=timesteps_input,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_ids,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]
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
