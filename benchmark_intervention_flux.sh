#!/bin/bash
# Benchmark inference with velocity intervention (VeloEdit)
#
# Generates benchmark outputs at multiple intervention strengths.
#
# Usage:
#    CUDA_DEVICE=3 bash benchmark_intervention_flux.sh
#   ./benchmark_intervention_flux.sh                              # Full run, default strengths
#   ./benchmark_intervention_flux.sh --test                       # Test run (5 images)
#   ./benchmark_intervention_flux.sh --gpu 0 --start 0 --end 175 # Batch processing
#   ./benchmark_intervention_flux.sh --preserve-steps 4 --edit-steps 1      # Separate region steps
#   ./benchmark_intervention_flux.sh --first-step-align-steps 8             # Align first step to 8-step schedule
#   ./benchmark_intervention_flux.sh --blend-weights "0.3,0.5,0.7"          # Multiple a values
#   ./benchmark_intervention_flux.sh --similarity-mode gt                   # Use benchmark GT mask
#   ./benchmark_intervention_flux.sh --overwrite-existing                   # Force regeneration
#   CHECK_METADATA=False ./benchmark_intervention_flux.sh                   # Resume without metadata config check

set -e

# ======== Default Configuration ========
CUDA_DEVICE=${CUDA_DEVICE:-0}
BENCHMARK_PATH="./benchmark"
OUTPUT_PATH="./benchmark_intervention_outputs_flux"

# Sampling (flux_config defaults)
NUM_INFERENCE_STEPS=30
GUIDANCE_SCALE=2.5
SEED=42
FIRST_STEP_ALIGN_STEPS=4

# Velocity intervention
PRESERVE_INTERVENTION_STEPS=4
EDIT_INTERVENTION_STEPS=4
SIMILARITY_THRESHOLD=0.8
# SIMILARITY_MODE="cosine"
# SIMILARITY_MODE="gt"
SIMILARITY_MODE="elementwise"
ENABLE_INTERV="yes"
BLEND_WEIGHTS="0.0,0.2,0.4,0.6,0.8"
# BLEND_WEIGHTS="0.0,0.05,0.25,0.7,0.75"
# BLEND_WEIGHTS="0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95"

# Model
LORA_PATH=""
MODEL_PATH=""

# Batch
START_IDX=""
END_IDX=""

# Output options
SAVE_MASKS=""
SAVE_HEATMAPS=""
SAVE_STEP_IMAGES=""
SAVE_ANALYSIS=""
OVERWRITE_EXISTING=""
CHECK_METADATA=${CHECK_METADATA:-False}

normalize_bool() {
    local name="$1"
    local value="$2"
    local normalized="${value,,}"

    case "$normalized" in
        true|1|yes|y|on)
            echo "True"
            ;;
        false|0|no|n|off)
            echo "False"
            ;;
        *)
            echo "Invalid $name: $value (expected True or False)" >&2
            exit 1
            ;;
    esac
}

# ======== Parse Arguments ========
while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            START_IDX=0
            END_IDX=5
            OUTPUT_PATH="./benchmark_intervention_outputs_test"
            shift
            ;;
        --gpu)
            CUDA_DEVICE="$2"
            shift 2
            ;;
        --start)
            START_IDX="$2"
            shift 2
            ;;
        --end)
            END_IDX="$2"
            shift 2
            ;;
        --output)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --benchmark)
            BENCHMARK_PATH="$2"
            shift 2
            ;;
        --steps)
            NUM_INFERENCE_STEPS="$2"
            shift 2
            ;;
        --guidance)
            GUIDANCE_SCALE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --first-step-align-steps)
            FIRST_STEP_ALIGN_STEPS="$2"
            shift 2
            ;;
        --lora)
            LORA_PATH="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --preserve-steps)
            PRESERVE_INTERVENTION_STEPS="$2"
            shift 2
            ;;
        --edit-steps)
            EDIT_INTERVENTION_STEPS="$2"
            shift 2
            ;;
        --threshold)
            SIMILARITY_THRESHOLD="$2"
            shift 2
            ;;
        --similarity-mode)
            SIMILARITY_MODE="$2"
            shift 2
            ;;
        --disable-interv)
            ENABLE_INTERV=""
            shift
            ;;
        --blend-weights)
            BLEND_WEIGHTS="$2"
            shift 2
            ;;
        --save-masks)
            SAVE_MASKS="yes"
            shift
            ;;
        --save-heatmaps)
            SAVE_HEATMAPS="yes"
            shift
            ;;
        --save-step-images)
            SAVE_STEP_IMAGES="yes"
            shift
            ;;
        --save-analysis)
            SAVE_ANALYSIS="yes"
            shift
            ;;
        --overwrite-existing)
            OVERWRITE_EXISTING="yes"
            shift
            ;;
        --disable-metadata-check)
            CHECK_METADATA="False"
            shift
            ;;
        --enable-metadata-check)
            CHECK_METADATA="True"
            shift
            ;;
        --save-all)
            SAVE_MASKS="yes"
            SAVE_HEATMAPS="yes"
            SAVE_STEP_IMAGES="yes"
            SAVE_ANALYSIS="yes"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Run benchmark inference with velocity intervention at multiple strengths."
            echo ""
            echo "Basic options:"
            echo "  --test                  Test run (5 images)"
            echo "  --gpu N                 CUDA device (default: $CUDA_DEVICE)"
            echo "  --start N               Start index (0-indexed)"
            echo "  --end N                 End index (exclusive)"
            echo "  --output PATH           Output directory"
            echo "  --benchmark PATH        Benchmark data directory"
            echo ""
            echo "Sampling options:"
            echo "  --steps N               Inference steps (default: $NUM_INFERENCE_STEPS)"
            echo "  --guidance F            Guidance scale (default: $GUIDANCE_SCALE)"
            echo "  --seed N                Random seed (default: $SEED)"
            echo "  --first-step-align-steps N  Align first step to N-step schedule when steps > N (default: $FIRST_STEP_ALIGN_STEPS; <=0 disables)"
            echo ""
            echo "Model options:"
            echo "  --lora PATH             LoRA weights path"
            echo "  --model-path PATH       Override model path"
            echo ""
            echo "Intervention options:"
            echo "  --preserve-steps N      Preserve/non-edit region intervention steps; negative means effective_steps+N (default: $PRESERVE_INTERVENTION_STEPS)"
            echo "  --edit-steps N          Edit region intervention steps; negative means effective_steps+N (default: $EDIT_INTERVENTION_STEPS)"
            echo "  --threshold F           Similarity threshold (default: $SIMILARITY_THRESHOLD)"
            echo "  --similarity-mode MODE  Similarity mode: elementwise, cosine, or gt (default: $SIMILARITY_MODE)"
            echo "  --disable-interv        Disable preserve replacement and edit blending"
            echo "  --blend-weights LIST    Comma-separated blend weights (a values) to benchmark (default: $BLEND_WEIGHTS)"
            echo ""
            echo "Output options:"
            echo "  --save-masks            Save similarity mask images"
            echo "  --save-heatmaps         Save similarity heatmap images"
            echo "  --save-step-images      Save per-step denoising images"
            echo "  --save-analysis         Save JSON/CSV analysis data"
            echo "  --save-all              Enable all save options"
            echo "  --overwrite-existing    Regenerate outputs instead of resuming from existing strength images"
            echo "  --disable-metadata-check  Set CHECK_METADATA=False"
            echo "  --enable-metadata-check   Set CHECK_METADATA=True (default)"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

CHECK_METADATA="$(normalize_bool CHECK_METADATA "$CHECK_METADATA")"

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

# ======== Print Configuration ========
echo "============================================"
echo " Benchmark Intervention - FLUX Kontext"
echo " (VeloEdit)"
echo "============================================"
echo "  CUDA Device:           $CUDA_DEVICE"
echo "  Benchmark Path:        $BENCHMARK_PATH"
echo "  Output Path:           $OUTPUT_PATH"
echo "  Inference Steps:       $NUM_INFERENCE_STEPS"
echo "  Guidance Scale:        $GUIDANCE_SCALE"
echo "  Seed:                  $SEED"
echo "  First-step Align:      $FIRST_STEP_ALIGN_STEPS"
echo "  ---"
echo "  Preserve Steps:        $PRESERVE_INTERVENTION_STEPS"
echo "  Edit Steps:            $EDIT_INTERVENTION_STEPS"
echo "  Similarity Threshold:  $SIMILARITY_THRESHOLD"
echo "  Similarity Mode:       $SIMILARITY_MODE"
echo "  Intervention:          $([ -n "$ENABLE_INTERV" ] && echo enabled || echo disabled)"
echo "  Blend Weights:         $BLEND_WEIGHTS"
[ -n "$LORA_PATH" ]      && echo "  LoRA:                  $LORA_PATH"
[ -n "$MODEL_PATH" ]     && echo "  Model Path:            $MODEL_PATH"
[ -n "$START_IDX" ]      && echo "  Start Index:           $START_IDX"
[ -n "$END_IDX" ]        && echo "  End Index:             $END_IDX"
echo "  ---"
[ -n "$SAVE_MASKS" ]       && echo "  Save Masks:            yes"
[ -n "$SAVE_HEATMAPS" ]    && echo "  Save Heatmaps:         yes"
[ -n "$SAVE_STEP_IMAGES" ] && echo "  Save Step Images:      yes"
[ -n "$SAVE_ANALYSIS" ]    && echo "  Save Analysis:         yes"
[ -n "$OVERWRITE_EXISTING" ] && echo "  Overwrite Existing:    yes"
echo "  Metadata Check:        $CHECK_METADATA"
echo "============================================"

# ======== Build Command ========
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

CMD="python3 ${SCRIPT_DIR}/benchmark_intervention_flux.py \
    --benchmark-path $BENCHMARK_PATH \
    --output-path $OUTPUT_PATH \
    --num-inference-steps $NUM_INFERENCE_STEPS \
    --guidance-scale $GUIDANCE_SCALE \
    --seed $SEED \
    --first-step-align-steps $FIRST_STEP_ALIGN_STEPS \
    --preserve-intervention-steps $PRESERVE_INTERVENTION_STEPS \
    --edit-intervention-steps $EDIT_INTERVENTION_STEPS \
    --similarity-threshold $SIMILARITY_THRESHOLD \
    --similarity-mode $SIMILARITY_MODE \
    --blend-weights $BLEND_WEIGHTS"

[ -z "$ENABLE_INTERV" ]    && CMD="$CMD --disable-interv"
[ -n "$LORA_PATH" ]        && CMD="$CMD --lora $LORA_PATH"
[ -n "$MODEL_PATH" ]       && CMD="$CMD --model-path $MODEL_PATH"
[ -n "$START_IDX" ]        && CMD="$CMD --start-idx $START_IDX"
[ -n "$END_IDX" ]          && CMD="$CMD --end-idx $END_IDX"
[ -n "$SAVE_MASKS" ]       && CMD="$CMD --save-masks"
[ -n "$SAVE_HEATMAPS" ]    && CMD="$CMD --save-heatmaps"
[ -n "$SAVE_STEP_IMAGES" ] && CMD="$CMD --save-step-images"
[ -n "$SAVE_ANALYSIS" ]    && CMD="$CMD --save-analysis"
[ -n "$OVERWRITE_EXISTING" ] && CMD="$CMD --overwrite-existing"
[ "$CHECK_METADATA" = "False" ] && CMD="$CMD --disable-metadata-check"

echo "Running: $CMD"
echo "============================================"

$CMD

echo "============================================"
echo " Benchmark intervention completed!"
echo " Results saved to: $OUTPUT_PATH"
echo "============================================"
