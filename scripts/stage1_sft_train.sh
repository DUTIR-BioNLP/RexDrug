#!/bin/bash
# ============================================================
# Stage 1: Supervised Fine-Tuning (SFT) Training
#
# This script trains the base model with SFT on drug combination
# extraction task using Chain-of-Thought (CoT) data.
# ============================================================

set -e

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# Parse arguments
MODEL_TYPE="${1:-llama}"      # llama or qwen
DATASET="${2:-drugcomb}"       # drugcomb or ddi
SEED="${3:-42}"
GPU="${4:-0}"

# Validate arguments
if [[ "$MODEL_TYPE" != "llama" && "$MODEL_TYPE" != "qwen" ]]; then
    echo "Usage: $0 <model_type> <dataset> [seed] [gpu]"
    echo "  model_type: llama or qwen"
    echo "  dataset: drugcomb or ddi"
    echo "  seed: random seed (default: 42)"
    echo "  gpu: GPU device ID (default: 0)"
    exit 1
fi

# Set model configuration
if [ "$MODEL_TYPE" == "llama" ]; then
    MODEL_PATH="${LLAMA_MODEL_PATH}"
    MODEL_NAME="llama3_1"
    TEMPLATE="llama3_2"
else
    MODEL_PATH="${QWEN_MODEL_PATH}"
    MODEL_NAME="qwen2_5"
    TEMPLATE="qwen2_5"
fi

# Set dataset paths
if [ "$DATASET" == "drugcomb" ]; then
    TRAIN_DATA="${DRUGCOMB_DIR}/sft_cot_data/drugcomb_train.jsonl"
else
    TRAIN_DATA="${DDI13_DIR}/sft_cot_data/train_ddi.jsonl"
fi

# Output directory
EXP_NAME="${MODEL_NAME}_sft_cot_${DATASET}_seed${SEED}"
OUTPUT_PATH="${OUTPUT_DIR}/stage1_sft/${EXP_NAME}"

# Print configuration
print_config
echo ""
echo "=== Stage 1: SFT Training ==="
echo "Model Type: ${MODEL_TYPE}"
echo "Model Path: ${MODEL_PATH}"
echo "Dataset: ${DATASET}"
echo "Train Data: ${TRAIN_DATA}"
echo "Output: ${OUTPUT_PATH}"
echo "Seed: ${SEED}"
echo "GPU: ${GPU}"
echo "=============================="

# Validate paths
if [ ! -f "${TRAIN_DATA}" ]; then
    echo "ERROR: Training data not found: ${TRAIN_DATA}"
    exit 1
fi

if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: Model not found: ${MODEL_PATH}"
    echo "Please update LLAMA_MODEL_PATH or QWEN_MODEL_PATH in scripts/config.sh"
    exit 1
fi

# Run SFT training
CUDA_VISIBLE_DEVICES=${GPU} swift sft \
    --model "${MODEL_PATH}" \
    --model_type "${MODEL_NAME}" \
    --template "${TEMPLATE}" \
    --output_dir "${OUTPUT_PATH}" \
    \
    --train_type lora \
    --lora_rank ${DEFAULT_LORA_RANK} \
    --lora_alpha ${DEFAULT_LORA_ALPHA} \
    --target_modules all-linear \
    \
    --dataset "${TRAIN_DATA}" \
    --num_train_epochs ${DEFAULT_SFT_EPOCHS} \
    --per_device_train_batch_size ${DEFAULT_SFT_BATCH_SIZE} \
    --learning_rate ${DEFAULT_SFT_LR} \
    --warmup_ratio 0.1 \
    \
    --save_strategy epoch \
    --logging_steps 20 \
    --gradient_checkpointing false \
    --eval_strategy no \
    \
    --seed ${SEED} \
    --data_seed ${SEED}

echo ""
echo "=== Stage 1 Completed ==="
echo "SFT checkpoint saved to: ${OUTPUT_PATH}"
echo ""
echo "Next step: Run stage2_merge_lora.sh to merge LoRA weights"
