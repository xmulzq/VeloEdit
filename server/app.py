
import io
import os
import sys
import base64
import argparse
import re
from pathlib import Path
from typing import Dict, List
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from VeloEdit.server.wrapper import create_analyzer, AnalyzerWrapper

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SERVER_CONFIG = {
    "host": "0.0.0.0",
    "port": 5001,
    "max_content_length": 100 * 1024 * 1024,
}

app = Flask(__name__, static_folder="static")
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = SERVER_CONFIG["max_content_length"]

loaded_analyzers: Dict[str, AnalyzerWrapper] = {}

def decode_image(data: str) -> Image.Image:
    if "," in data:
        data = data.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(data)))


def encode_image(image: Image.Image, fmt: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def sanitize_filename(text: str) -> str:
    text = re.sub(r'[<>:"/\\|?*]', '_', text)
    if len(text) > 100:
        text = text[:100]
    return text.strip()


def save_images_to_disk(
    images: List[Image.Image],
    prompt: str,
    strength_values: List[float],
    save_mode: str = "single",
    model_type: str = "flux"
) -> List[str]:
    saved_paths = []
    clean_prompt = sanitize_filename(prompt)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if model_type in ["qwen", "qwen_single"]:
        model_dir = RESULTS_DIR / "qwen"
    else:
        model_dir = RESULTS_DIR / "flux"

    model_dir.mkdir(exist_ok=True)

    if save_mode == "single" and len(images) == 1:
        strength = strength_values[0] if strength_values else 0
        filename = f"{clean_prompt}_strength{strength:.2f}_{timestamp}.png"
        filepath = model_dir / filename
        images[0].save(filepath)
        saved_paths.append(str(filepath))
        print(f"Saved single image: {filepath}")

    elif save_mode == "concat" and len(images) > 1:
        min_strength = min(strength_values) if strength_values else 0
        max_strength = max(strength_values) if strength_values else 0

        widths, heights = zip(*(img.size for img in images))
        total_width = sum(widths)
        max_height = max(heights)

        concat_image = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for img in images:
            concat_image.paste(img, (x_offset, 0))
            x_offset += img.width

        filename = f"{clean_prompt}_strength{min_strength:.2f}-{max_strength:.2f}_{timestamp}.png"
        filepath = model_dir / filename
        concat_image.save(filepath)
        saved_paths.append(str(filepath))
        print(f"Saved concatenated image: {filepath}")

    elif save_mode == "multiple" and len(images) > 1:
        subfolder = model_dir / f"{clean_prompt}_{timestamp}"
        subfolder.mkdir(exist_ok=True)

        for img, strength in zip(images, strength_values):
            filename = f"{clean_prompt}_strength{strength:.2f}.png"
            filepath = subfolder / filename
            img.save(filepath)
            saved_paths.append(str(filepath))

        print(f"Saved {len(images)} images to: {subfolder}")

    return saved_paths


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "loaded_analyzers": list(loaded_analyzers.keys())
    })


@app.route("/models", methods=["GET"])
def get_models():
    loaded = list(loaded_analyzers.keys())
    frontend_loaded = []
    for model in loaded:
        if model == "qwen_single":
            frontend_loaded.append("qwen")
        else:
            frontend_loaded.append(model)

    return jsonify({
        "available": ["flux", "qwen"],
        "loaded": list(set(frontend_loaded)),
    })


@app.route("/save", methods=["POST"])
def save():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400

        images_data = data.get("images", [])
        prompt = data.get("prompt", "output")
        strength_values = data.get("strength_values", [])
        save_mode = data.get("save_mode", "multiple")
        model_type = data.get("model_type", "flux")

        if not images_data:
            return jsonify({"success": False, "error": "No images provided"}), 400

        images = [decode_image(img_data).convert("RGB") for img_data in images_data]

        if not strength_values:
            strength_values = [i / len(images) for i in range(len(images))]

        if save_mode == "concat":
            saved_paths = save_images_to_disk(images, prompt, strength_values, "concat", model_type)
        elif save_mode == "reverse_concat":
            reversed_images = list(reversed(images))
            reversed_strengths = list(reversed(strength_values))
            saved_paths = save_images_to_disk(reversed_images, f"{prompt}_reversed", reversed_strengths, "concat", model_type)
        elif save_mode == "multiple":
            saved_paths = save_images_to_disk(images, prompt, strength_values, "multiple", model_type)
        else:
            return jsonify({"success": False, "error": f"Invalid save_mode: {save_mode}"}), 400

        return jsonify({
            "success": True,
            "saved_paths": saved_paths,
            "message": f"Saved {len(saved_paths)} file(s)"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400

        image_data = data.get("image")
        prompt = data.get("prompt")
        model_type = data.get("model")
        image_name = data.get("image_name", "input")

        num_inference_steps = data.get("num_inference_steps", 15)
        seed = data.get("seed", 42)
        intervention_steps = data.get("intervention_steps", 0)
        similarity_threshold = data.get("similarity_threshold", 0.8)
        enable_blend = data.get("enable_blend", False)
        blend_weight = data.get("blend_weight", 0.5)

        if not all([image_data, prompt, model_type]):
            return jsonify({
                "success": False,
                "error": "Missing required fields: image, prompt, model"
            }), 400

        actual_model_type = model_type
        if model_type == "qwen" and model_type not in loaded_analyzers:
            if "qwen_single" in loaded_analyzers:
                actual_model_type = "qwen_single"
            else:
                return jsonify({
                    "success": False,
                    "error": f"Model '{model_type}' not loaded. Available: {list(loaded_analyzers.keys())}"
                }), 400
        elif actual_model_type not in loaded_analyzers:
            return jsonify({
                "success": False,
                "error": f"Model '{model_type}' not loaded. Available: {list(loaded_analyzers.keys())}"
            }), 400

        input_image = decode_image(image_data).convert("RGB")

        intervention_config = {
            "num_inference_steps": num_inference_steps,
            "seed": seed,
            "intervention_steps": intervention_steps,
            "similarity_threshold": similarity_threshold,
            "enable_blend": enable_blend,
            "blend_weight": blend_weight,
        }

        analyzer = loaded_analyzers[actual_model_type]
        result = analyzer.analyze(input_image, prompt, image_name, intervention_config)

        response = {
            "success": True,
            "model_name": result.model_name,
            "num_steps": result.num_steps,
            "summary": result.summary,
            "interventions_applied": result.interventions_applied,
            "per_step": [
                {
                    "step": d.step,
                    "sigma": d.sigma,
                    "preserve_magnitude": d.preserve_magnitude,
                    "edit_magnitude": d.edit_magnitude,
                    "total_magnitude": d.total_magnitude,
                    "preserve_ratio": d.preserve_ratio,
                    "edit_ratio": d.edit_ratio,
                    "angle_to_reference": d.angle_to_reference,
                    "distance_to_ref": d.distance_to_ref,
                }
                for d in result.decompositions
            ],
        }

        if result.generated_image is not None:
            response["generated_image"] = encode_image(result.generated_image)

        if result.step_images:
            response["step_images"] = [encode_image(img) for img in result.step_images]

        if result.similarity_mask_images:
            response["similarity_mask_images"] = [
                encode_image(img) for img in result.similarity_mask_images
            ]

        if result.similarity_heatmap_images:
            response["similarity_heatmap_images"] = [
                encode_image(img) for img in result.similarity_heatmap_images
            ]

        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


def parse_args():
    parser = argparse.ArgumentParser(description="Velocity Analysis Server")
    parser.add_argument(
        "--models", nargs="+", required=True,
        help="Models to load (e.g., flux:0 qwen:0,1 qwen_single:2)"
    )
    parser.add_argument("--port", type=int, default=5001, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--lora", type=str, default=None, help="LoRA path for FLUX")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 50)
    print("Velocity Analysis Server (Refactored)")
    print("=" * 50)

    for model_spec in args.models:
        model_type = model_spec.split(":")[0]
        print(f"Loading {model_spec}...")
        wrapper = create_analyzer(model_spec, lora_path=args.lora if model_type == "flux" else None)
        loaded_analyzers[model_type] = wrapper
        print(f"  {model_type} loaded successfully")

    print("=" * 50)
    print(f"Server starting on http://{args.host}:{args.port}")
    print(f"Loaded models: {list(loaded_analyzers.keys())}")
    print("=" * 50)

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
