
from ml_collections import ConfigDict


def flux_config() -> ConfigDict:
    config = ConfigDict()

    config.model = ConfigDict()
    config.model.name = "flux"
    config.model.path = "black-forest-labs/FLUX.1-Kontext-dev"
    config.model.dtype = "bfloat16"

    config.sampling = ConfigDict()
    config.sampling.num_inference_steps = 30
    config.sampling.guidance_scale = 2.5
    # Match the first denoising delta to a short reference schedule for benchmark parity.
    config.sampling.first_step_align_steps = 4

    config.output = ConfigDict()
    config.output.output_dir = "./velocity_output"
    config.output.dpi = 150

    return config


def qwen_config() -> ConfigDict:
    config = ConfigDict()

    config.model = ConfigDict()
    config.model.name = "qwen"
    config.model.path = "Qwen/Qwen-Image-Edit-2509"
    config.model.dtype = "bfloat16"

    config.sampling = ConfigDict()
    config.sampling.num_inference_steps = 28
    config.sampling.guidance_scale = 1.0
    config.sampling.true_cfg_scale = 4.0
    # Qwen is especially sensitive to the first step, so keep this enabled by default.
    config.sampling.first_step_align_steps = 6

    config.output = ConfigDict()
    config.output.output_dir = "./velocity_output"
    config.output.dpi = 150

    return config


def get_config(name: str) -> ConfigDict:
    configs = {
        "flux": flux_config,
        "qwen": qwen_config,
    }

    if name not in configs:
        raise ValueError(f"Unknown config: {name}. Available: {list(configs.keys())}")

    return configs[name]()
