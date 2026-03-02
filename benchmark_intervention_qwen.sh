#!/bin/bash
# Benchmark inference with velocity intervention for Qwen-Image-Edit-2509
#
# Generates benchmark outputs at multiple intervention strengths.
# Each strength produces a separate output subdirectory (int0, int2, int4, ...).
#
# Usage:
#   ./benchmark_intervention_qwen.sh                              # Full run (single GPU)
#   ./benchmark_intervention_qwen.sh --test                       # Test run (5 images)
#   ./benchmark_intervention_qwen.sh --gpu 0 --start 0 --end 175  # Single GPU batch
#   ./benchmark_intervention_qwen.sh --gpu 0,1 --multi-gpu        # Multi-GPU mode (recommended for Qwen)
#   ./benchmark_intervention_qwen.sh --gpu 2,3 --multi-gpu --test # Multi-GPU test
#   ./benchmark_intervention_qwen.sh --int-steps "0,3,6"          # Custom strengths
#   ./benchmark_intervention_qwen.sh --blend --blend-weights "0.3,0.5,0.7"  # Multiple a values
#
# Note: Qwen model is large (~7B+), multi-GPU mode (--multi-gpu) with 2+ GPUs is recommended.

set -e

# ======== Default Configuration ========
# Default: use GPU 0,1 for multi-GPU mode (Qwen needs ~2 GPUs)
CUDA_DEVICE=${CUDA_DEVICE:-"2,3"}
BENCHMARK_PATH="./benchmark"
OUTPUT_PATH="./benchmark_intervention_outputs_qwen"

# Sampling (qwen_config defaults)
NUM_INFERENCE_STEPS=6
GUIDANCE_SCALE=1.0
TRUE_CFG_SCALE=4.0
SEED=42

# Velocity intervention
INTERVENTION_STEPS="6"
SIMILARITY_THRESHOLD=0.8
ENABLE_BLEND=""
BLEND_WEIGHTS="0.0,0.2,0.4,0.6,0.8"

# Model
MODEL_PATH=""
# Default to multi-GPU mode for Qwen (model is too large for single GPU)
MULTI_GPU="yes"

# Batch
START_IDX=""
END_IDX=""

# Output options
SAVE_MASKS=""
SAVE_HEATMAPS=""
SAVE_STEP_IMAGES=""
SAVE_ANALYSIS=""

# ======== Parse Arguments ========
while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            START_IDX=0
            END_IDX=5
            OUTPUT_PATH="./benchmark_intervention_outputs_qwen_test"
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
        --true-cfg)
            TRUE_CFG_SCALE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --multi-gpu)
            MULTI_GPU="yes"
            shift
            ;;
        --single-gpu)
            MULTI_GPU=""
            shift
            ;;
        --int-steps)
            INTERVENTION_STEPS="$2"
            shift 2
            ;;
        --threshold)
            SIMILARITY_THRESHOLD="$2"
            shift 2
            ;;
        --blend)
            ENABLE_BLEND="yes"
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
            echo "Run benchmark inference with velocity intervention for Qwen-Image-Edit-2509."
            echo ""
            echo "Basic options:"
            echo "  --test                  Test run (5 images)"
            echo "  --gpu DEVICES           CUDA device(s), e.g., '0' or '0,1' (default: $CUDA_DEVICE)"
            echo "  --start N               Start index (0-indexed)"
            echo "  --end N                 End index (exclusive)"
            echo "  --output PATH           Output directory"
            echo "  --benchmark PATH        Benchmark data directory"
            echo ""
            echo "Sampling options:"
            echo "  --steps N               Inference steps (default: $NUM_INFERENCE_STEPS)"
            echo "  --guidance F            Guidance scale (default: $GUIDANCE_SCALE)"
            echo "  --true-cfg F            True CFG scale (default: $TRUE_CFG_SCALE)"
            echo "  --seed N                Random seed (default: $SEED)"
            echo ""
            echo "Model options:"
            echo "  --model-path PATH       Override model path"
            echo "  --multi-gpu             Use device_map='balanced' for multi-GPU (default: enabled)"
            echo "  --single-gpu            Disable multi-GPU mode (requires 80GB+ GPU)"
            echo ""
            echo "Intervention options:"
            echo "  --int-steps LIST        Comma-separated intervention step counts (default: $INTERVENTION_STEPS)"
            echo "  --threshold F           Similarity threshold (default: $SIMILARITY_THRESHOLD)"
            echo "  --blend                 Enable blending for low similarity elements"
            echo "  --blend-weights LIST    Comma-separated blend weights (a values) to benchmark (default: $BLEND_WEIGHTS)"
            echo ""
            echo "Output options:"
            echo "  --save-masks            Save similarity mask images"
            echo "  --save-heatmaps         Save similarity heatmap images"
            echo "  --save-step-images      Save per-step denoising images"
            echo "  --save-analysis         Save JSON/CSV analysis data"
            echo "  --save-all              Enable all save options"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

# ======== Print Configuration ========
echo "============================================"
echo " Benchmark Intervention - Qwen-Image-Edit-2509"
echo " (VeloEdit)"
echo "============================================"
echo "  CUDA Device:           $CUDA_DEVICE"
echo "  Benchmark Path:        $BENCHMARK_PATH"
echo "  Output Path:           $OUTPUT_PATH"
echo "  Inference Steps:       $NUM_INFERENCE_STEPS"
echo "  Guidance Scale:        $GUIDANCE_SCALE"
echo "  True CFG Scale:        $TRUE_CFG_SCALE"
echo "  Seed:                  $SEED"
echo "  ---"
echo "  Intervention Steps:    $INTERVENTION_STEPS"
echo "  Similarity Threshold:  $SIMILARITY_THRESHOLD"
[ -n "$ENABLE_BLEND" ]   && echo "  Blend:                 enabled"
echo "  Blend Weights:         $BLEND_WEIGHTS"
[ -n "$MODEL_PATH" ]     && echo "  Model Path:            $MODEL_PATH"
[ -n "$MULTI_GPU" ]      && echo "  Multi-GPU:             enabled"
[ -n "$START_IDX" ]      && echo "  Start Index:           $START_IDX"
[ -n "$END_IDX" ]        && echo "  End Index:             $END_IDX"
echo "  ---"
[ -n "$SAVE_MASKS" ]       && echo "  Save Masks:            yes"
[ -n "$SAVE_HEATMAPS" ]    && echo "  Save Heatmaps:         yes"
[ -n "$SAVE_STEP_IMAGES" ] && echo "  Save Step Images:      yes"
[ -n "$SAVE_ANALYSIS" ]    && echo "  Save Analysis:         yes"
echo "============================================"

# ======== Build Command ========
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

CMD="python3 ${SCRIPT_DIR}/benchmark_intervention_qwen.py \
    --benchmark-path $BENCHMARK_PATH \
    --output-path $OUTPUT_PATH \
    --num-inference-steps $NUM_INFERENCE_STEPS \
    --guidance-scale $GUIDANCE_SCALE \
    --true-cfg-scale $TRUE_CFG_SCALE \
    --seed $SEED \
    --intervention-steps $INTERVENTION_STEPS \
    --similarity-threshold $SIMILARITY_THRESHOLD \
    --blend-weights $BLEND_WEIGHTS"

[ -n "$ENABLE_BLEND" ]     && CMD="$CMD --enable-blend"
[ -n "$MODEL_PATH" ]       && CMD="$CMD --model-path $MODEL_PATH"
[ -n "$MULTI_GPU" ]        && CMD="$CMD --multi-gpu"
[ -n "$START_IDX" ]        && CMD="$CMD --start-idx $START_IDX"
[ -n "$END_IDX" ]          && CMD="$CMD --end-idx $END_IDX"
[ -n "$SAVE_MASKS" ]       && CMD="$CMD --save-masks"
[ -n "$SAVE_HEATMAPS" ]    && CMD="$CMD --save-heatmaps"
[ -n "$SAVE_STEP_IMAGES" ] && CMD="$CMD --save-step-images"
[ -n "$SAVE_ANALYSIS" ]    && CMD="$CMD --save-analysis"

echo "Running: $CMD"
echo "============================================"

$CMD

echo "============================================"
echo " Benchmark intervention completed!"
echo " Results saved to: $OUTPUT_PATH"
echo "============================================"
