#!/bin/bash
# ============================================================
# Stage 3: GRPO (Group Relative Policy Optimization) Training
#
# This script performs GRPO reinforcement learning training
# using the merged SFT model from Stage 2.
# ============================================================

set -e

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# Parse arguments
MERGED_MODEL="${1}"
MODEL_TYPE="${2:-llama}"      # llama or qwen
DATASET="${3:-drugcomb}"       # drugcomb or ddi
SEED="${4:-42}"
GPU="${5:-0}"

if [ -z "$MERGED_MODEL" ]; then
    echo "Usage: $0 <merged_model_path> <model_type> <dataset> [seed] [gpu]"
    echo ""
    echo "Arguments:"
    echo "  merged_model_path: Path to the merged model from Stage 2"
    echo "  model_type: llama or qwen"
    echo "  dataset: drugcomb or ddi"
    echo "  seed: random seed (default: 42)"
    echo "  gpu: GPU device ID (default: 0)"
    echo ""
    echo "Example:"
    echo "  $0 outputs/stage2_merged/merged_model llama drugcomb 42 0"
    exit 1
fi

# Validate merged model
if [ ! -d "$MERGED_MODEL" ]; then
    echo "ERROR: Merged model not found: $MERGED_MODEL"
    exit 1
fi

# Set model configuration
if [ "$MODEL_TYPE" == "llama" ]; then
    MODEL_NAME="llama3_1"
    TEMPLATE="llama3_2"
else
    MODEL_NAME="qwen2_5"
    TEMPLATE="qwen2_5"
fi

# Set dataset and reward functions
if [ "$DATASET" == "drugcomb" ]; then
    TRAIN_DATA="${DRUGCOMB_DIR}/grpo_cot_data_v2/train_drugcomb.jsonl"
    REWARD_FUNCS="drugcomb_cot_format drugcomb_cot_think drugcomb_coverage_cot drugcomb_accuracy_cot"
    REWARD_WEIGHTS="0.05 0.05 0.1 0.8"
else
    TRAIN_DATA="${DDI13_DIR}/grpo_cot_data/train_ddi.jsonl"
    REWARD_FUNCS="ddi_cot_format ddi_cot_think ddi_coverage_cot ddi_accuracy_cot"
    REWARD_WEIGHTS="0.05 0.05 0.1 0.8"
fi

# Output directory
EXP_NAME="${MODEL_NAME}_grpo_cot_${DATASET}_seed${SEED}"
OUTPUT_PATH="${OUTPUT_DIR}/stage3_grpo/${EXP_NAME}"

# Print configuration
print_config
echo ""
echo "=== Stage 3: GRPO Training ==="
echo "Merged Model: ${MERGED_MODEL}"
echo "Model Type: ${MODEL_TYPE}"
echo "Dataset: ${DATASET}"
echo "Train Data: ${TRAIN_DATA}"
echo "Reward Functions: ${REWARD_FUNCS}"
echo "Reward Weights: ${REWARD_WEIGHTS}"
echo "Output: ${OUTPUT_PATH}"
echo "Seed: ${SEED}"
echo "GPU: ${GPU}"
echo "==============================="

# Validate paths
if [ ! -f "${TRAIN_DATA}" ]; then
    echo "ERROR: Training data not found: ${TRAIN_DATA}"
    exit 1
fi

if [ ! -f "${PLUGIN_PATH}" ]; then
    echo "ERROR: GRPO reward plugin not found: ${PLUGIN_PATH}"
    exit 1
fi

# Run GRPO training
CUDA_VISIBLE_DEVICES=${GPU} swift rlhf \
    --rlhf_type grpo \
    --model "${MERGED_MODEL}" \
    --model_type "${MODEL_NAME}" \
    --template "${TEMPLATE}" \
    \
    --dataset "${TRAIN_DATA}" \
    --split_dataset_ratio 0 \
    \
    --external_plugins "${PLUGIN_PATH}" \
    --reward_funcs ${REWARD_FUNCS} \
    --reward_weights ${REWARD_WEIGHTS} \
    \
    --train_type lora \
    --lora_rank ${DEFAULT_LORA_RANK} \
    --lora_alpha ${DEFAULT_LORA_ALPHA} \
    --target_modules all-linear \
    \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --sleep_level 0 \
    \
    --torch_dtype bfloat16 \
    --max_completion_length 2048 \
    \
    --num_train_epochs ${DEFAULT_GRPO_EPOCHS} \
    --per_device_train_batch_size ${DEFAULT_GRPO_BATCH_SIZE} \
    --gradient_accumulation_steps 4 \
    --learning_rate ${DEFAULT_GRPO_LR} \
    --warmup_ratio 0.1 \
    \
    --num_generations 8 \
    --temperature 0.7 \
    --top_p 0.9 \
    --beta 0.04 \
    \
    --save_strategy epoch \
    --logging_steps 1 \
    --output_dir "${OUTPUT_PATH}" \
    \
    --report_to tensorboard \
    --log_completions true \
    --seed ${SEED}

echo ""
echo "=== Stage 3 Completed ==="
echo "GRPO checkpoint saved to: ${OUTPUT_PATH}"
echo ""
echo "Next step: Run inference.sh to evaluate the trained model"
