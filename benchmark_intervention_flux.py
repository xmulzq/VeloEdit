"""
Benchmark inference with velocity intervention for FLUX Kontext pipeline.

For each benchmark image, generates outputs at multiple intervention strengths,
allowing systematic comparison of how velocity intervention affects editing.

Uses FLUXVelocityAnalyzer for deterministic sampling with intervention support.
"""

import torch
import os
import sys
import json
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from velocity_refactor import FLUXVelocityAnalyzer, get_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark inference with velocity intervention (velocity_refactor)"
    )

    # Model
    parser.add_argument("--model-path", type=str, default=None,
                        help="Override model path (default: from flux_config)")
    parser.add_argument("--lora", type=str, default=None,
                        help="LoRA weights path (safetensors file or PEFT directory)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="CUDA device")

    # Sampling
    parser.add_argument("--num-inference-steps", type=int, default=None,
                        help="Inference steps (default: from config)")
    parser.add_argument("--guidance-scale", type=float, default=None,
                        help="Guidance scale (default: from config)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Velocity intervention
    parser.add_argument("--intervention-steps", type=str, default="0,2,4,6",
                        help="Comma-separated list of intervention step counts to benchmark")
    parser.add_argument("--similarity-threshold", type=float, default=0.8,
                        help="Similarity threshold for intervention")
    parser.add_argument("--no-blend", action="store_true",
                        help="Disable blending for low similarity elements (blend is ON by default)")
    parser.add_argument("--blend-weights", type=str, default="0.5",
                        help="Comma-separated list of blend weights to benchmark")

    # Data
    parser.add_argument("--benchmark-path", type=str,
                        default="/hpfs/jerry/RL4Beauty/kontinuouskontext/benchmark",
                        help="Path to benchmark directory")
    parser.add_argument("--output-path", type=str,
                        default="./benchmark_intervention_outputs",
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

    return parser.parse_args()


def parse_intervention_steps(steps_str):
    """Parse comma-separated intervention steps into sorted int list."""
    return sorted(set(int(s.strip()) for s in steps_str.split(",")))


def parse_blend_weights(weights_str):
    """Parse comma-separated blend weights into sorted float list."""
    return sorted(set(float(w.strip()) for w in weights_str.split(",")))


def strength_to_filename(strength: float) -> str:
    """Convert strength value to filename format: strength_X_XX.jpg

    Examples:
        0.0  -> strength_0_00.jpg
        0.2  -> strength_0_20.jpg
        1.0  -> strength_1_00.jpg
    """
    # Format as X_XX (e.g., 0_20 for 0.2)
    s = f"{strength:.2f}".replace(".", "_")
    return f"strength_{s}.jpg"


def run_benchmark(args, analyzer):
    """Run benchmark with multiple intervention strengths.

    Output structure matches benchmark_outputs:
        {output_path}/{total_steps}_{int_steps}_{threshold}/
            {editing_type_dir}/
                {image_id}/
                    original.jpg
                    strength_0_00.jpg  (strength = 1 - a)
                    strength_0_20.jpg
                    ...
                    metadata.json
    """
    config = get_config("flux")

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
    enable_blend = not args.no_blend
    threshold = args.similarity_threshold

    # Parse intervention strengths and blend weights to benchmark
    intervention_steps_list = parse_intervention_steps(args.intervention_steps)
    blend_weights_list = parse_blend_weights(args.blend_weights)
    print(f"[Config] Intervention steps to benchmark: {intervention_steps_list}")
    print(f"[Config] Blend weights (a values) to benchmark: {blend_weights_list}")
    print(f"[Config] Corresponding strengths (1-a): {[round(1-a, 2) for a in blend_weights_list]}")
    print(f"[Config] similarity_threshold={threshold}, enable_blend={enable_blend}")

    # For each intervention_steps, we'll create a separate top-level directory
    # and run all blend_weights for each image within that directory
    for int_steps in intervention_steps_list:
        # Build top-level directory: {total_steps}_{int_steps}_{threshold}
        top_dir_name = f"{num_steps}_{int_steps}_{threshold}"
        top_dir = os.path.join(args.output_path, top_dir_name)
        os.makedirs(top_dir, exist_ok=True)

        print(f"\n[Config] Processing int_steps={int_steps}, output: {top_dir}")
        print(f"[Config] Will generate {len(blend_weights_list)} strength files per image")

        for idx, image_id in enumerate(tqdm(image_ids, desc=f"int_steps={int_steps}")):
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

            # Build output directory: {top_dir}/{image_path_dir}/{image_id}/
            # image_rel_path is like "0_random_140/000000000000.jpg"
            # We want: {top_dir}/0_random_140/000000000000/
            image_path_dir = os.path.dirname(image_rel_path)  # e.g., "0_random_140"
            output_dir = os.path.join(top_dir, image_path_dir, image_id)
            os.makedirs(output_dir, exist_ok=True)

            # Save original (once per image)
            orig_path = os.path.join(output_dir, "original.jpg")
            if not os.path.exists(orig_path):
                orig_image.save(orig_path)

            # Metadata to accumulate results from all blend_weights
            metadata = {
                "image_id": image_id,
                "original_path": image_rel_path,
                "editing_instruction": instruction,
                "editing_type_id": entry.get("editing_type_id", ""),
                "num_inference_steps": num_steps,
                "seed": seed,
                "intervention_steps": int_steps,
                "similarity_threshold": threshold,
                "enable_blend": enable_blend,
                "strengths": {},  # strength -> result info
            }

            # Run each blend_weight and save as strength_{1-a}.jpg
            for bw in blend_weights_list:
                strength = round(1.0 - bw, 2)  # strength = 1 - a

                intervention_config = {
                    "num_inference_steps": num_steps,
                    "seed": seed,
                    "intervention_steps": int_steps,
                    "similarity_threshold": threshold,
                    "enable_blend": enable_blend,
                    "blend_weight": bw,
                }

                result = analyzer.analyze(
                    image=orig_image,
                    prompt=instruction,
                    image_path=full_image_path,
                    intervention_config=intervention_config,
                )

                # Save generated image as strength_X_XX.jpg
                if result.generated_image:
                    strength_filename = strength_to_filename(strength)
                    result.generated_image.save(os.path.join(output_dir, strength_filename))

                # Record in metadata
                metadata["strengths"][f"{strength:.2f}"] = {
                    "blend_weight_a": bw,
                    "interventions_applied": result.interventions_applied,
                }

                # Optional: save masks per strength
                if args.save_masks and result.similarity_mask_images:
                    masks_dir = os.path.join(output_dir, f"masks_s{strength:.2f}")
                    os.makedirs(masks_dir, exist_ok=True)
                    for i, img in enumerate(result.similarity_mask_images):
                        img.save(os.path.join(masks_dir, f"mask_{i:02d}.png"))

                # Optional: save heatmaps per strength
                if args.save_heatmaps and result.similarity_heatmap_images:
                    heatmaps_dir = os.path.join(output_dir, f"heatmaps_s{strength:.2f}")
                    os.makedirs(heatmaps_dir, exist_ok=True)
                    for i, img in enumerate(result.similarity_heatmap_images):
                        img.save(os.path.join(heatmaps_dir, f"heatmap_{i:02d}.png"))

                # Optional: save step images per strength
                if args.save_step_images and result.step_images:
                    steps_dir = os.path.join(output_dir, f"steps_s{strength:.2f}")
                    os.makedirs(steps_dir, exist_ok=True)
                    for i, img in enumerate(result.step_images):
                        img.save(os.path.join(steps_dir, f"step_{i:02d}.png"))

                torch.cuda.empty_cache()

            # Save metadata (once per image, after all strengths)
            with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)

            # Optional: save analysis JSON/CSV (aggregate)
            if args.save_analysis:
                from velocity_refactor.output import export_to_json
                export_to_json(metadata, os.path.join(output_dir, "analysis.json"))

            if (idx + 1) % 50 == 0:
                print(f"[Progress] {idx + 1}/{len(image_ids)} images processed")

    print(f"\n[Done] Results saved to: {args.output_path}")


def main():
    args = parse_args()
    print(f"[Args] {args}")

    # Build analyzer
    config = get_config("flux")
    if args.model_path:
        config.model.path = args.model_path
    if args.num_inference_steps:
        config.sampling.num_inference_steps = args.num_inference_steps
    if args.guidance_scale:
        config.sampling.guidance_scale = args.guidance_scale

    analyzer = FLUXVelocityAnalyzer(
        config=config,
        device=args.device,
        lora_path=args.lora,
    )

    print("[Model] Loading pipeline...")
    analyzer.load_model()
    analyzer.model_loaded = True
    print("[Model] Pipeline ready.")

    run_benchmark(args, analyzer)


if __name__ == "__main__":
    main()
