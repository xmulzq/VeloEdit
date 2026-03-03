# VeloEdit

VeloEdit is a tool for analyzing and intervening in the velocity field of diffusion models, supporting FLUX.1-Kontext and Qwen-Image-Edit models. By decomposing velocity vectors into preserve and edit components, VeloEdit enables precise control over the image editing process for more controllable image generation. You can access the project at [VeloEdit](https://xmulzq.github.io/VeloEdit).

## Core Features

- **Velocity Field Decomposition**: Decomposes diffusion model velocity vectors into preserve and edit components
- **Element-wise Intervention**: Precise velocity field intervention based on similarity thresholds to control image editing strength
- **Multi-model Support**: Supports FLUX.1-Kontext and Qwen-Image-Edit-2509 models
- **Visualization Analysis**: Generates step-by-step predicted images, similarity masks, and heatmaps
- **Web Interface**: Provides an interactive web server with real-time parameter adjustment and batch processing
- **LoRA Support**: FLUX model supports loading LoRA weights

## Project Structure

```
VeloEdit/
├── core/                    # Core algorithms
│   ├── sampler.py          # Deterministic sampler (Euler method)
│   ├── intervention.py     # Velocity field intervention logic
│   ├── decomposer.py       # Velocity vector decomposition
│   └── types.py            # Data type definitions
├── analyzers/              # Model analyzers
│   ├── base.py            # Base analyzer class
│   ├── flux.py            # FLUX model analyzer
│   └── qwen.py            # Qwen model analyzer
├── server/                 # Web server
│   ├── app.py             # Flask application
│   ├── wrapper.py         # Analyzer wrapper
│   └── static/            # Frontend interface
│       └── index.html
├── config/                 # Configuration management
│   └── config.py          # Model configuration
├── scripts/                # Utility scripts
│   └── analyze.py         # Command-line analysis tool
├── output/                 # Output module
│   └── exporters.py       # Result export
|── benchmark_intervention_flux.py
|── benchmark_intervention_flux.sh      # Benchmark script
|── benchmark_intervention_qwen.py
└── benchmark_intervention_qwen.sh      # Benchmark script
```

## Installation

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### 1. Web Server Mode

Launch the interactive web interface:

```bash
# Load FLUX model only (GPU 0)
python -m VeloEdit.server.app --models flux:0 --port 5001

# Load both FLUX and Qwen models (multi-GPU)
python -m VeloEdit.server.app --models flux:0 qwen:0,1 --port 5001

# Load FLUX model with LoRA
python -m VeloEdit.server.app --models flux:0 --port 5001 --lora /path/to/lora.safetensors
```

Access `http://localhost:5001` to open the web interface.

#### API Endpoints

- `GET /health` - Health check
- `GET /models` - Get available model list
- `POST /analyze` - Analyze image and generate results
- `POST /save_images` - Save generated images

### 2. Command-line Mode

Use scripts for batch analysis:

```bash
python scripts/analyze.py \
  --model flux \
  --image input.jpg \
  --prompt "Add a hat" \
  ...
```

### 3. Benchmark

Run PIEbench benchmark:

```bash
bash benchmark_infer.sh 
```
