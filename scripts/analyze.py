
import argparse
import os
from pathlib import Path
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from VeloEdit import (
    FLUXVelocityAnalyzer,
    QwenVelocityAnalyzer,
    get_config,
)
from VeloEdit.output import export_to_json, export_to_csv


def parse_args():
    parser = argparse.ArgumentParser(description="Velocity Analysis CLI")

    parser.add_argument("--model", type=str, required=True, choices=["flux", "qwen"],
                        help="Model to use")
    parser.add_argument("--image", type=str, required=True,
                        help="Input image path")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Edit prompt")

    parser.add_argument("--model-path", type=str, default=None,
                        help="Custom model path (overrides config)")
    parser.add_argument("--lora", type=str, default=None,
                        help="LoRA weights path (FLUX only)")
    parser.add_argument("--multi-gpu", action="store_true",
                        help="Use multi-GPU mode for Qwen")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use (single GPU mode)")

    parser.add_argument("--steps", type=int, default=15,
                        help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    parser.add_argument("--intervention-steps", type=int, default=0,
                        help="Number of intervention steps (0 = disabled)")
    parser.add_argument("--similarity-threshold", type=float, default=0.8,
                        help="Similarity threshold for intervention")
    parser.add_argument("--enable-blend", action="store_true",
                        help="Enable blending for low similarity elements")
    parser.add_argument("--blend-weight", type=float, default=0.5,
                        help="Blend weight (a) for reference velocity")

    parser.add_argument("--output-dir", type=str, default="./velocity_output",
                        help="Output directory")
    parser.add_argument("--save-images", action="store_true",
                        help="Save step images and masks")

    return parser.parse_args()


def main():
    args = parse_args()

    config = get_config(args.model)

    if args.model_path:
        config.model.path = args.model_path
    config.sampling.num_inference_steps = args.steps

    if args.model == "flux":
        analyzer = FLUXVelocityAnalyzer(
            config=config,
            device=args.device,
            lora_path=args.lora,
        )
    else:
        analyzer = QwenVelocityAnalyzer(
            config=config,
            device=args.device,
            multi_gpu=args.multi_gpu,
        )

    print(f"[Input] Loading image: {args.image}")
    image = Image.open(args.image).convert("RGB")

    intervention_config = {
        "num_inference_steps": args.steps,
        "seed": args.seed,
        "intervention_steps": args.intervention_steps,
        "similarity_threshold": args.similarity_threshold,
        "enable_blend": args.enable_blend,
        "blend_weight": args.blend_weight,
    }

    print(f"[Analyze] Running analysis with {args.model} model...")
    result = analyzer.analyze(
        image=image,
        prompt=args.prompt,
        image_path=args.image,
        intervention_config=intervention_config,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(args.image).stem
    export_to_json(result, str(output_dir / f"{base_name}_analysis.json"))
    export_to_csv(result, str(output_dir / f"{base_name}_analysis.csv"))

    if result.generated_image:
        result.generated_image.save(output_dir / f"{base_name}_generated.png")
        print(f"[Output] Generated image saved")

    if args.save_images:
        if result.step_images:
            steps_dir = output_dir / "steps"
            steps_dir.mkdir(exist_ok=True)
            for i, img in enumerate(result.step_images):
                img.save(steps_dir / f"step_{i:02d}.png")
            print(f"[Output] Step images saved to {steps_dir}")

        if result.similarity_mask_images:
            masks_dir = output_dir / "masks"
            masks_dir.mkdir(exist_ok=True)
            for i, img in enumerate(result.similarity_mask_images):
                img.save(masks_dir / f"mask_{i:02d}.png")
            print(f"[Output] Mask images saved to {masks_dir}")

        if result.similarity_heatmap_images:
            heatmaps_dir = output_dir / "heatmaps"
            heatmaps_dir.mkdir(exist_ok=True)
            for i, img in enumerate(result.similarity_heatmap_images):
                img.save(heatmaps_dir / f"heatmap_{i:02d}.png")
            print(f"[Output] Heatmap images saved to {heatmaps_dir}")

    print("\n" + "=" * 50)
    print("Analysis Summary")
    print("=" * 50)
    summary = result.summary
    print(f"Model: {result.model_name}")
    print(f"Steps: {result.num_steps}")
    print(f"Interventions Applied: {result.interventions_applied}")
    print(f"Mean Preserve Ratio: {summary.get('mean_preserve_ratio', 0):.4f}")
    print(f"Mean Edit Ratio: {summary.get('mean_edit_ratio', 0):.4f}")
    print(f"Mean Angle: {summary.get('mean_angle', 0):.2f}°")
    print(f"Early Preserve Ratio: {summary.get('early_preserve_ratio', 0):.4f}")
    print(f"Late Preserve Ratio: {summary.get('late_preserve_ratio', 0):.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
