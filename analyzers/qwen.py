
from typing import Dict, Any, Optional, Tuple, List
import torch
import numpy as np
from PIL import Image
from ml_collections import ConfigDict

from .base import BaseVelocityAnalyzer
from ..core.sampler import align_first_step_to_reference_step, log_sampling_schedule


CONDITION_IMAGE_SIZE = 384 * 384
VAE_IMAGE_SIZE = 1024 * 1024


class QwenVelocityAnalyzer(BaseVelocityAnalyzer):
    """Qwen-Image-Edit adapter with optional balanced multi-GPU loading."""

    def __init__(
        self,
        config: ConfigDict,
        device: str = "cuda",
        save_tensors: bool = False,
        multi_gpu: bool = False,
    ):
        super().__init__(config, device, save_tensors)
        self.multi_gpu = multi_gpu

    def load_model(self) -> None:
        try:
            from diffusers import QwenImageEditPlusPipeline
        except ImportError:
            raise ImportError(
                "QwenImageEditPlusPipeline not found. "
                "Please ensure you have the correct version of diffusers installed."
            )

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }

        config_dtype = self.config.model.dtype
        dtype = dtype_map.get(config_dtype, torch.bfloat16)
        print(f"[Qwen] Using dtype: {dtype}")

        print(f"[Qwen] Loading model from {self.config.model.path}...")

        if self.multi_gpu:
            print("[Qwen] Using device_map='balanced' for multi-GPU")
            self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
                self.config.model.path,
                torch_dtype=dtype,
                device_map="balanced",
            )
        else:
            print(f"[Qwen] Using single GPU: {self.device}")
            self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
                self.config.model.path,
                torch_dtype=dtype,
            )
            self.pipeline.to(self.device)

        print("[Qwen] Model loaded successfully.")

    def _get_execution_device(self) -> torch.device:
        if self.multi_gpu:
            # In device_map mode diffusers tracks the active execution device internally.
            return self.pipeline._execution_device
        return torch.device(self.device)

    def _calculate_dimensions(
        self,
        max_area: int,
        aspect_ratio: float,
    ) -> Tuple[int, int]:
        """Calculate Qwen-friendly dimensions while preserving aspect ratio."""
        width = np.sqrt(max_area * aspect_ratio)
        height = width / aspect_ratio

        width = round(width / 32) * 32
        height = round(height / 32) * 32

        return width, height

    def _get_qwen_prompt_embeds(
        self,
        prompt: str,
        image: Optional[List[Image.Image]] = None,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = device or self._get_execution_device()
        dtype = dtype or self.pipeline.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
        if isinstance(image, list):
            base_img_prompt = ""
            for i, img in enumerate(image):
                base_img_prompt += img_prompt_template.format(i + 1)
        elif image is not None:
            base_img_prompt = img_prompt_template.format(1)
        else:
            base_img_prompt = ""

        # Qwen text encoding expects image placeholders to be embedded in the prompt.
        template = self.pipeline.prompt_template_encode
        drop_idx = self.pipeline.prompt_template_encode_start_idx
        txt = [template.format(base_img_prompt + e) for e in prompt]

        model_inputs = self.pipeline.processor(
            text=txt,
            images=image,
            padding=True,
            return_tensors="pt",
        ).to(device)

        outputs = self.pipeline.text_encoder(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            pixel_values=model_inputs.get("pixel_values"),
            image_grid_thw=model_inputs.get("image_grid_thw"),
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states[-1]
        split_hidden_states = self.pipeline._extract_masked_hidden(
            hidden_states,
            model_inputs["attention_mask"],
        )
        # Remove template prefix tokens so sampler conditioning aligns with edit text.
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max(e.size(0) for e in split_hidden_states)

        # Pad variable-length prompt embeddings because transformer forward expects a batch tensor.
        prompt_embeds = torch.stack([
            torch.cat([
                u,
                u.new_zeros(max_seq_len - u.size(0), u.size(1))
            ])
            for u in split_hidden_states
        ])

        encoder_attention_mask = torch.stack([
            torch.cat([
                u,
                u.new_zeros(max_seq_len - u.size(0))
            ])
            for u in attn_mask_list
        ])

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        if encoder_attention_mask.all():
            encoder_attention_mask = None

        return prompt_embeds, encoder_attention_mask

    def _prepare_inputs(
        self,
        image: Image.Image,
        prompt: str,
        num_inference_steps: int,
        seed: int,
    ) -> Dict[str, Any]:
        try:
            from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import retrieve_timesteps
            from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import calculate_shift
        except ImportError:
            raise ImportError("Cannot import Qwen pipeline functions. Check diffusers version.")

        device = self._get_execution_device()
        batch_size = 1
        negative_prompt = " "

        original_width, original_height = image.size
        aspect_ratio = original_width / original_height

        width, height = self._calculate_dimensions(
            VAE_IMAGE_SIZE,
            aspect_ratio,
        )

        multiple_of = self.pipeline.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of

        # Qwen uses a lower-resolution image for visual-language conditioning and
        # a separate 1024^2-area image for VAE latents.
        condition_width, condition_height = self._calculate_dimensions(
            CONDITION_IMAGE_SIZE, aspect_ratio
        )

        print(f"[Qwen] Original size: {original_width}x{original_height}, Working size: {width}x{height}")

        condition_image = self.pipeline.image_processor.resize(image, condition_height, condition_width)
        condition_images = [condition_image]

        vae_image = self.pipeline.image_processor.preprocess(image, height, width).unsqueeze(2)
        vae_images = [vae_image]

        input_image_resized = self.pipeline.image_processor.resize(image, height, width)

        prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(
            prompt=prompt,
            image=condition_images,
            device=device,
        )

        true_cfg_scale = getattr(self.config.sampling, "true_cfg_scale", 4.0)
        do_true_cfg = true_cfg_scale > 1 and negative_prompt is not None

        # True CFG needs a separate unconditional text/image encoding path.
        negative_prompt_embeds = None
        negative_prompt_embeds_mask = None
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self._get_qwen_prompt_embeds(
                prompt=negative_prompt,
                image=condition_images,
                device=device,
            )

        num_channels_latents = self.pipeline.transformer.config.in_channels // 4

        # Match diffusers examples/tests that pass torch.manual_seed(seed), which
        # is a CPU generator and produces different latents from a CUDA generator.
        generator = torch.Generator(device="cpu").manual_seed(seed)
        print(f"[Seed] Using random seed: {seed}")

        latents, image_latents = self.pipeline.prepare_latents(
            vae_images,
            batch_size,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            None,
        )

        img_shapes = [
            [
                (1, height // self.pipeline.vae_scale_factor // 2, width // self.pipeline.vae_scale_factor // 2),
                (1, height // self.pipeline.vae_scale_factor // 2, width // self.pipeline.vae_scale_factor // 2),
            ]
        ] * batch_size

        transformer_dtype = self.pipeline.transformer.dtype
        latents = latents.to(dtype=transformer_dtype)
        if image_latents is not None:
            image_latents = image_latents.to(dtype=transformer_dtype)

        # Use the encoded source-image latent as the intervention reference.
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

        # Log the effective schedule because Qwen outputs are sensitive to the first step.
        log_sampling_schedule(
            "Qwen",
            requested_num_inference_steps,
            raw_sigma_schedule,
            sigma_schedule,
            reference_sigma_schedule,
            first_step_align_steps,
        )

        guidance_scale = self.config.sampling.guidance_scale
        if self.pipeline.transformer.config.guidance_embeds and guidance_scale is None:
            raise ValueError("guidance_scale is required for guidance-distilled model.")
        elif self.pipeline.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        def v_pred_fn(z, sigma):
            # Concatenate generated and condition-image tokens exactly as the pipeline does.
            latent_model_input = z
            if image_latents is not None:
                latent_model_input = torch.cat([z, image_latents], dim=1)

            train_timesteps = self.pipeline.scheduler.config.get("num_train_timesteps", 1000)
            sigma_tensor = torch.as_tensor(sigma, device=z.device, dtype=torch.float32)
            # Qwen transformer receives normalized timesteps derived from sigma.
            timesteps_input = (
                (sigma_tensor * train_timesteps)
                .expand(latent_model_input.shape[0])
                .to(dtype=z.dtype)
                / train_timesteps
            )

            with self.pipeline.transformer.cache_context("cond"):
                noise_pred = self.pipeline.transformer(
                    hidden_states=latent_model_input,
                    timestep=timesteps_input,
                    guidance=guidance,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes,
                    attention_kwargs={},
                    return_dict=False,
                )[0]

            # Only generated-token velocity participates in sampling/intervention.
            noise_pred = noise_pred[:, : latents.size(1)]

            if do_true_cfg:
                with self.pipeline.transformer.cache_context("uncond"):
                    noise_pred_negative = self.pipeline.transformer(
                        hidden_states=latent_model_input,
                        timestep=timesteps_input,
                        guidance=guidance,
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_hidden_states_mask=negative_prompt_embeds_mask,
                        img_shapes=img_shapes,
                        attention_kwargs={},
                        return_dict=False,
                    )[0]
                noise_pred_negative = noise_pred_negative[:, : latents.size(1)]

                comb_pred = noise_pred_negative + true_cfg_scale * (noise_pred - noise_pred_negative)

                # Rescale true-CFG output to preserve the conditional prediction norm.
                cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                noise_pred = comb_pred * (cond_norm / noise_norm)

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
            'input_image_resized': input_image_resized,
        }

    def _decode_latents(
        self,
        latents: torch.Tensor,
        height: int,
        width: int,
        original_height: int = None,
        original_width: int = None,
    ) -> Image.Image:
        device = self._get_execution_device()

        vae_dtype = self.pipeline.vae.dtype
        latents = latents.to(device=device, dtype=vae_dtype)

        # Qwen latents are packed token sequences before VAE decoding.
        latents = self.pipeline._unpack_latents(
            latents, height, width, self.pipeline.vae_scale_factor
        )

        # Apply Qwen VAE normalization constants before decoding.
        latents_mean = (
            torch.tensor(self.pipeline.vae.config.latents_mean)
            .view(1, self.pipeline.vae.config.z_dim, 1, 1, 1)
            .to(device, vae_dtype)
        )
        latents_std = 1.0 / torch.tensor(self.pipeline.vae.config.latents_std).view(
            1, self.pipeline.vae.config.z_dim, 1, 1, 1
        ).to(device, vae_dtype)

        for_vae = latents / latents_std + latents_mean

        with torch.no_grad():
            decoded = self.pipeline.vae.decode(for_vae, return_dict=False)[0]

        if decoded.dim() == 5:
            decoded = decoded.squeeze(2)

        image = self.pipeline.image_processor.postprocess(
            decoded, output_type="pil"
        )[0]

        if original_width is not None and original_height is not None:
            if image.size != (original_width, original_height):
                image = image.resize((original_width, original_height), Image.LANCZOS)

        return image
