
from typing import Optional
from PIL import Image

from ..analyzers import FLUXVelocityAnalyzer, QwenVelocityAnalyzer
from ..config import get_config
from ..core.types import AnalysisResult


class AnalyzerWrapper:

    def __init__(self, analyzer, model_type: str):
        self.analyzer = analyzer
        self.model_type = model_type

    def analyze(
        self,
        image: Image.Image,
        prompt: str,
        image_name: str = "",
        intervention_config: Optional[dict] = None,
    ) -> AnalysisResult:
        return self.analyzer.analyze(
            image=image,
            prompt=prompt,
            image_path=image_name,
            intervention_config=intervention_config,
        )


def create_analyzer(
    model_spec: str,
    lora_path: Optional[str] = None,
) -> AnalyzerWrapper:
    parts = model_spec.split(":")
    model_type = parts[0]
    gpu_ids = parts[1] if len(parts) > 1 else "0"

    if "," in gpu_ids:
        device = f"cuda:{gpu_ids.split(',')[0]}"
        multi_gpu = True
    else:
        device = f"cuda:{gpu_ids}"
        multi_gpu = False

    print(f"[Wrapper] Creating {model_type} analyzer on {device} (multi_gpu={multi_gpu})")

    if model_type in ["qwen", "qwen_single"]:
        config = get_config("qwen")
    else:
        config = get_config(model_type)

    if model_type == "flux":
        analyzer = FLUXVelocityAnalyzer(
            config=config,
            device=device,
            lora_path=lora_path,
        )
    elif model_type in ["qwen", "qwen_single"]:
        use_multi_gpu = multi_gpu if model_type == "qwen" else False
        analyzer = QwenVelocityAnalyzer(
            config=config,
            device=device,
            multi_gpu=use_multi_gpu,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    analyzer.load_model()
    analyzer.model_loaded = True

    return AnalyzerWrapper(analyzer, model_type)
