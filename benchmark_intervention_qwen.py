"""
Benchmark inference with velocity intervention for Qwen-Image-Edit-2509 pipeline.

For each benchmark image, generates outputs at multiple intervention strengths,
allowing systematic comparison of how velocity intervention affects editing.

Uses QwenVelocityAnalyzer for deterministic sampling with intervention support.
"""

import torch
import os
import sys
import json
import argparse
import math
import numpy as np
from pathlib import Path
from typing import List
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from VeloEdit import QwenVelocityAnalyzer, get_config
from VeloEdit.core.image_compare import save_compare_artifacts
from VeloEdit.core.sampler import align_first_step_to_reference_step


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark inference with velocity intervention for Qwen (velocity_refactor)"
    )

    # Model
    parser.add_argument("--model-path", type=str, default=None,
                        help="Override model path (default: from qwen_config)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="CUDA device")
    parser.add_argument("--multi-gpu", action="store_true",
                        help="Use device_map='balanced' for multi-GPU")

    # Sampling
    parser.add_argument("--num-inference-steps", type=int, default=None,
                        help="Inference steps (default: from config)")
    parser.add_argument("--guidance-scale", type=float, default=None,
                        help="Guidance scale (default: from config)")
    parser.add_argument("--true-cfg-scale", type=float, default=None,
                        help="True CFG scale for Qwen (default: from config)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--first-step-align-steps", type=int, default=None,
                        help="Reference step count for first-step alignment; default from config (6), <=0 disables")

    # Velocity intervention
    parser.add_argument("--preserve-intervention-steps", type=int, default=6,
                        help="Intervention steps for preserve/non-edit region; negative values mean effective_steps + value")
    parser.add_argument("--edit-intervention-steps", type=int, default=6,
                        help="Intervention steps for edit region; negative values mean effective_steps + value")
    parser.add_argument("--similarity-threshold", type=float, default=0.8,
                        help="Similarity threshold for intervention")
    parser.add_argument("--similarity-mode", type=str, default="elementwise",
                        choices=["elementwise", "cosine", "gt"],
                        help="Similarity mode for mask generation and intervention; gt uses mapping_file.json masks")
    parser.add_argument("--disable-interv", action="store_true",
                        help="Disable preserve replacement and edit blending")
    parser.add_argument("--blend-weights", type=str, default="0.5",
                        help="Comma-separated list of blend weights to benchmark")

    # Data
    parser.add_argument("--benchmark-path", type=str,
                        default="./benchmark",
                        help="Path to benchmark directory")
    parser.add_argument("--output-path", type=str,
                        default="./benchmark_intervention_outputs_qwen",
                        help="Output directory")

    # Batch processing
    parser.add_argument("--start-idx", type=int, default=None,
                        help="Start index (0-indexed)")
    parser.add_argument("--end-idx", type=int, default=None,
                        help="End index (exclusive)")

    # Output options
    parser.add_argument("--save-masks", action="store_true",
                        help="Save similarity mask images per step")
    parser.add_argument("--save-heatmaps", action="store_true",
                        help="Save similarity heatmap images per step")
    parser.add_argument("--save-step-images", action="store_true",
                        help="Save per-step denoising images")
    parser.add_argument("--save-analysis", action="store_true",
                        help="Save JSON/CSV analysis data per image")
    parser.add_argument("--overwrite-existing", action="store_true",
                        help="Regenerate outputs even when strength images already exist")
    parser.add_argument("--disable-metadata-check", action="store_true",
                        help="Resume without checking whether existing metadata matches the current config")

    return parser.parse_args()


def parse_blend_weights(weights_str):
    """Parse comma-separated blend weights into sorted float list."""
    return sorted(set(float(w.strip()) for w in weights_str.split(",")))


def get_mask_partition_name(similarity_mode: str) -> str:
    """Return the mask partition label used in output directory names."""
    if similarity_mode == "elementwise":
        return ""
    return f"mask_{similarity_mode}"


def strength_to_filename(strength: float) -> str:
    """Convert strength value to filename format: strength_X_XX.jpg"""
    s = f"{strength:.2f}".replace(".", "_")
    return f"strength_{s}.jpg"


def save_compare_if_complete(output_dir: str, blend_weights: List[float]) -> None:
    expected_paths = [
        Path(output_dir) / strength_to_filename(round(1.0 - bw, 2))
        for bw in blend_weights
    ]
    if not all(path.exists() for path in expected_paths):
        missing = [str(path.name) for path in expected_paths if not path.exists()]
        print(f"[Compare] Skip incomplete folder {output_dir}, missing: {missing}")
        return

    saved_paths = save_compare_artifacts(expected_paths, output_dir)
    print(f"[Compare] Saved: {', '.join(saved_paths)}")


def is_valid_image(path: str) -> bool:
    if not os.path.exists(path):
        return False

    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception as exc:
        print(f"[Resume] Existing image is invalid, will regenerate: {path} ({exc})")
        return False


def load_existing_metadata(output_dir: str) -> dict:
    metadata_path = os.path.join(output_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        return {}

    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        return metadata if isinstance(metadata, dict) else {}
    except Exception as exc:
        print(f"[Resume] Cannot read existing metadata, will rebuild: {metadata_path} ({exc})")
        return {}


def metadata_config_matches(existing_metadata: dict, current_metadata: dict) -> bool:
    if not existing_metadata:
        return True

    keys = [
        "num_inference_steps",
        "seed",
        "first_step_align_steps",
        "preserve_intervention_steps",
        "edit_intervention_steps",
        "similarity_threshold",
        "similarity_mode",
        "enable_interv",
        "model",
    ]
    return all(existing_metadata.get(key) == current_metadata.get(key) for key in keys)


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def calculate_qwen_working_size(image: Image.Image) -> tuple[int, int]:
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height
    width = math.sqrt((1024 ** 2) * aspect_ratio)
    height = width / aspect_ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    multiple_of = 16
    width = width // multiple_of * multiple_of
    height = height // multiple_of * multiple_of
    return width, height


def estimate_effective_steps_for_image(
    scheduler,
    image: Image.Image,
    requested_steps: int,
    first_step_align_steps: int,
) -> int:
    width, height = calculate_qwen_working_size(image)
    image_seq_len = (height // 16) * (width // 16)
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )

    requested_sigmas = np.linspace(1.0, 1 / requested_steps, requested_steps)
    scheduler.set_timesteps(
        requested_steps,
        device="cpu",
        sigmas=requested_sigmas,
        mu=mu,
    )
    raw_sigma_schedule = scheduler.sigmas.float().clone()

    if first_step_align_steps <= 0 or requested_steps <= first_step_align_steps:
        return len(raw_sigma_schedule) - 1

    reference_sigmas = np.linspace(
        1.0,
        1 / first_step_align_steps,
        first_step_align_steps,
    )
    scheduler.set_timesteps(
        first_step_align_steps,
        device="cpu",
        sigmas=reference_sigmas,
        mu=mu,
    )
    reference_sigma_schedule = scheduler.sigmas.float().clone()
    effective_sigma_schedule = align_first_step_to_reference_step(
        raw_sigma_schedule,
        reference_sigma_schedule,
        first_step_align_steps,
    )
    return len(effective_sigma_schedule) - 1


def format_effective_steps_label(effective_steps_by_image: dict) -> str:
    unique_steps = sorted(set(effective_steps_by_image.values()))
    if not unique_steps:
        return "unknown"
    if len(unique_steps) == 1:
        return str(unique_steps[0])
    if len(unique_steps) <= 3:
        return "mix" + "-".join(str(step) for step in unique_steps)
    return f"mix{unique_steps[0]}-{unique_steps[-1]}"


def estimate_effective_steps_by_image(
    benchmark_path: str,
    mapping_data: dict,
    image_ids: List[str],
    requested_steps: int,
    first_step_align_steps: int,
    model_path: str,
) -> dict:
    """Estimate per-image effective steps without loading the full Qwen pipeline."""
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_path,
        subfolder="scheduler",
    )
    effective_steps_by_image = {}
    for image_id in image_ids:
        image_rel_path = mapping_data[image_id]["image_path"]
        full_image_path = os.path.join(
            benchmark_path,
            "annotation_images",
            image_rel_path,
        )
        if not os.path.exists(full_image_path):
            continue
        with Image.open(full_image_path) as image:
            effective_steps_by_image[image_id] = estimate_effective_steps_for_image(
                scheduler,
                image,
                requested_steps,
                first_step_align_steps,
            )
    return effective_steps_by_image


def run_benchmark(args, analyzer):
    """Run benchmark with multiple intervention strengths."""
    config = get_config("qwen")

    # Load mapping file
    mapping_path = os.path.join(args.benchmark_path, "mapping_file.json")
    with open(mapping_path, 'r') as f:
        mapping_data = json.load(f)

    image_ids = sorted(mapping_data.keys())
    total = len(image_ids)

    # Batch slicing
    start = args.start_idx if args.start_idx is not None else 0
    end = args.end_idx if args.end_idx is not None else total
    image_ids = image_ids[start:end]
    print(f"[Benchmark] Processing images {start} to {end} ({len(image_ids)} images)")

    # Inference parameters
    num_steps = args.num_inference_steps or config.sampling.num_inference_steps
    seed = args.seed
    first_step_align_steps = args.first_step_align_steps
    if first_step_align_steps is None:
        first_step_align_steps = getattr(config.sampling, "first_step_align_steps", 6)
    preserve_steps = args.preserve_intervention_steps
    edit_steps = args.edit_intervention_steps
    enable_interv = not args.disable_interv
    threshold = args.similarity_threshold
    similarity_mode = args.similarity_mode

    # Parse blend weights to benchmark
    blend_weights_list = parse_blend_weights(args.blend_weights)
    # Output directory names need effective step counts before expensive generation starts.
    effective_steps_by_image = estimate_effective_steps_by_image(
        args.benchmark_path,
        mapping_data,
        image_ids,
        num_steps,
        first_step_align_steps,
        analyzer.config.model.path,
    )
    effective_steps_label = format_effective_steps_label(effective_steps_by_image)
    print(f"[Config] Preserve steps: {preserve_steps}")
    print(f"[Config] Edit steps: {edit_steps}")
    print(f"[Config] First-step alignment reference steps: {first_step_align_steps}")
    print(f"[Config] Effective sampling steps: {effective_steps_label}")
    print(f"[Config] Blend weights (a values) to benchmark: {blend_weights_list}")
    print(f"[Config] Corresponding strengths (1-a): {[round(1-a, 2) for a in blend_weights_list]}")
    print(
        f"[Config] similarity_threshold={threshold}, "
        f"similarity_mode={similarity_mode}, enable_interv={enable_interv}"
    )

    mask_partition_name = get_mask_partition_name(similarity_mode)
    mask_partition_suffix = f"_{mask_partition_name}" if mask_partition_name else ""
    top_dir_name = (
        f"{num_steps}_eff{effective_steps_label}_align{first_step_align_steps}"
        f"_pres{preserve_steps}_edit{edit_steps}{mask_partition_suffix}_{threshold}"
    )
    top_dir = os.path.join(args.output_path, top_dir_name)
    os.makedirs(top_dir, exist_ok=True)

    print(f"\n[Config] Output: {top_dir}")
    print(f"[Config] Will generate {len(blend_weights_list)} strength files per image")
    print(f"[Config] Resume: {'disabled, overwrite existing outputs' if args.overwrite_existing else 'enabled, skip existing valid strength images'}")
    check_metadata = not args.disable_metadata_check
    print(f"[Config] Metadata check: {'enabled' if check_metadata else 'disabled'}")

    for idx, image_id in enumerate(tqdm(image_ids, desc=f"pres={preserve_steps},edit={edit_steps}")):
        entry = mapping_data[image_id]
        image_rel_path = entry['image_path']
        instruction = entry['editing_instruction']

        full_image_path = os.path.join(
            args.benchmark_path, "annotation_images", image_rel_path
        )
        if not os.path.exists(full_image_path):
            print(f"[Warning] Image not found: {full_image_path}")
            continue

        orig_image = Image.open(full_image_path).convert("RGB")

        # Build output directory
        image_path_dir = os.path.dirname(image_rel_path)
        output_dir = os.path.join(top_dir, image_path_dir, image_id)
        os.makedirs(output_dir, exist_ok=True)

        # Save original
        orig_path = os.path.join(output_dir, "original.jpg")
        if not os.path.exists(orig_path):
            orig_image.save(orig_path)

        existing_metadata = load_existing_metadata(output_dir)

        metadata = {
            "image_id": image_id,
            "original_path": image_rel_path,
            "editing_instruction": instruction,
            "editing_type_id": entry.get("editing_type_id", ""),
            "num_inference_steps": num_steps,
            "effective_num_inference_steps": effective_steps_by_image.get(image_id),
            "seed": seed,
            "first_step_align_steps": first_step_align_steps,
            "preserve_intervention_steps": preserve_steps,
            "edit_intervention_steps": edit_steps,
            "similarity_threshold": threshold,
            "similarity_mode": similarity_mode,
            "enable_interv": enable_interv,
            "model": "Qwen-Image-Edit-2509",
            "strengths": {},
        }
        if mask_partition_name:
            metadata["mask_partition_mode"] = mask_partition_name
        if similarity_mode == "gt":
            metadata["gt_mask_source"] = "mapping_file.json"
        can_resume_existing = (
            not check_metadata
            or metadata_config_matches(existing_metadata, metadata)
        )
        if can_resume_existing:
            existing_strengths = existing_metadata.get("strengths", {})
            metadata["strengths"] = existing_strengths if isinstance(existing_strengths, dict) else {}
        else:
            print(f"[Resume] Existing metadata config differs, regenerating strengths in {output_dir}")

        for bw in blend_weights_list:
            # Public benchmark strength is the amount left from model prediction.
            strength = round(1.0 - bw, 2)
            strength_key = f"{strength:.2f}"
            strength_filename = strength_to_filename(strength)
            strength_path = os.path.join(output_dir, strength_filename)

            if (
                can_resume_existing
                and not args.overwrite_existing
                and is_valid_image(strength_path)
            ):
                print(f"[Resume] Skip existing: {strength_path}")
                metadata["strengths"].setdefault(
                    strength_key,
                    {
                        "blend_weight_a": bw,
                        "effective_num_inference_steps": effective_steps_by_image.get(image_id),
                        "status": "existing",
                        "resumed": True,
                    },
                )
                continue

            intervention_config = {
                "num_inference_steps": num_steps,
                "seed": seed,
                "first_step_align_steps": first_step_align_steps,
                "preserve_intervention_steps": preserve_steps,
                "edit_intervention_steps": edit_steps,
                "similarity_threshold": threshold,
                "similarity_mode": similarity_mode,
                "enable_interv": enable_interv,
                "blend_weight": bw,
            }
            if similarity_mode == "gt":
                # PIE-style GT masks are stored as RLE arrays in mapping_file.json.
                intervention_config["gt_mask"] = entry.get("mask")
                intervention_config["gt_mask_size"] = (512, 512)

            result = analyzer.analyze(
                image=orig_image,
                prompt=instruction,
                image_path=full_image_path,
                intervention_config=intervention_config,
            )

            if result.generated_image:
                result.generated_image.save(strength_path)

            metadata["strengths"][strength_key] = {
                "blend_weight_a": bw,
                "effective_num_inference_steps": result.num_steps,
                "interventions_applied": result.interventions_applied,
                "preserve_interventions_applied": result.preserve_interventions_applied,
                "edit_interventions_applied": result.edit_interventions_applied,
                "status": "generated",
            }

            if args.save_masks and result.similarity_mask_images:
                masks_dir = os.path.join(output_dir, f"masks_s{strength:.2f}")
                os.makedirs(masks_dir, exist_ok=True)
                for i, img in enumerate(result.similarity_mask_images):
                    img.save(os.path.join(masks_dir, f"mask_{i:02d}.png"))

            if args.save_heatmaps and result.similarity_heatmap_images:
                heatmaps_dir = os.path.join(output_dir, f"heatmaps_s{strength:.2f}")
                os.makedirs(heatmaps_dir, exist_ok=True)
                for i, img in enumerate(result.similarity_heatmap_images):
                    img.save(os.path.join(heatmaps_dir, f"heatmap_{i:02d}.png"))

            if args.save_step_images and result.step_images:
                steps_dir = os.path.join(output_dir, f"steps_s{strength:.2f}")
                os.makedirs(steps_dir, exist_ok=True)
                for i, img in enumerate(result.step_images):
                    img.save(os.path.join(steps_dir, f"step_{i:02d}.png"))

            torch.cuda.empty_cache()

        with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)

        if args.save_analysis:
            from VeloEdit.output import export_to_json
            export_to_json(metadata, os.path.join(output_dir, "analysis.json"))

        save_compare_if_complete(output_dir, blend_weights_list)

        if (idx + 1) % 50 == 0:
            print(f"[Progress] {idx + 1}/{len(image_ids)} images processed")

    print(f"\n[Done] Results saved to: {args.output_path}")


def main():
    args = parse_args()
    print(f"[Args] {args}")

    config = get_config("qwen")
    if args.model_path:
        config.model.path = args.model_path
    if args.num_inference_steps:
        config.sampling.num_inference_steps = args.num_inference_steps
    if args.guidance_scale:
        config.sampling.guidance_scale = args.guidance_scale
    if args.true_cfg_scale:
        config.sampling.true_cfg_scale = args.true_cfg_scale
    if args.first_step_align_steps is not None:
        config.sampling.first_step_align_steps = args.first_step_align_steps

    analyzer = QwenVelocityAnalyzer(
        config=config,
        device=args.device,
        multi_gpu=args.multi_gpu,
    )

    run_benchmark(args, analyzer)


if __name__ == "__main__":
    main()
