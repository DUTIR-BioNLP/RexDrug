#!/bin/bash
# ============================================================
# Full Pipeline: Two-Stage Drug Combination Extraction
#
# This script runs the complete pipeline:
#   Stage 1: SFT Training
#   Stage 2: Merge LoRA
#   Stage 3: GRPO Training
#   Stage 4: Inference & Evaluation
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

echo "=============================================="
echo "Two-Stage Drug Combination Extraction Pipeline"
echo "=============================================="
echo "Model Type: ${MODEL_TYPE}"
echo "Dataset:    ${DATASET}"
echo "Seed:       ${SEED}"
echo "GPU:        ${GPU}"
echo "=============================================="

# Validate configuration
print_config
validate_paths || exit 1

# Set model name
if [ "$MODEL_TYPE" == "llama" ]; then
    MODEL_NAME="llama3_1"
else
    MODEL_NAME="qwen2_5"
fi

# Define paths
SFT_OUTPUT="${OUTPUT_DIR}/stage1_sft/${MODEL_NAME}_sft_cot_${DATASET}_seed${SEED}"
MERGED_OUTPUT="${OUTPUT_DIR}/stage2_merged/${MODEL_NAME}_${DATASET}_seed${SEED}"
GRPO_OUTPUT="${OUTPUT_DIR}/stage3_grpo/${MODEL_NAME}_grpo_cot_${DATASET}_seed${SEED}"

# ============================================================
# Stage 1: SFT Training
# ============================================================
echo ""
echo "######################################"
echo "# Stage 1: SFT Training              #"
echo "######################################"
echo ""

bash "${SCRIPT_DIR}/stage1_sft_train.sh" "${MODEL_TYPE}" "${DATASET}" "${SEED}" "${GPU}"

# Find best checkpoint (last epoch)
# ms-swift creates: output_dir/v0-YYYYMMDD-HHMMSS/checkpoint-*
SFT_VERSION_DIR=$(ls -d ${SFT_OUTPUT}/v*-* 2>/dev/null | sort -V | tail -1)
if [ -z "$SFT_VERSION_DIR" ]; then
    echo "ERROR: No SFT version directory found in ${SFT_OUTPUT}"
    echo "Expected format: v0-YYYYMMDD-HHMMSS"
    exit 1
fi
SFT_CHECKPOINT=$(ls -d ${SFT_VERSION_DIR}/checkpoint-* 2>/dev/null | sort -V | tail -1)
if [ -z "$SFT_CHECKPOINT" ]; then
    echo "ERROR: No SFT checkpoint found in ${SFT_VERSION_DIR}"
    exit 1
fi
echo "Using SFT checkpoint: ${SFT_CHECKPOINT}"

# ============================================================
# Stage 2: Merge LoRA
# ============================================================
echo ""
echo "######################################"
echo "# Stage 2: Merge LoRA Weights        #"
echo "######################################"
echo ""

bash "${SCRIPT_DIR}/stage2_merge_lora.sh" "${SFT_CHECKPOINT}" "${MODEL_NAME}_${DATASET}_seed${SEED}"

# ============================================================
# Stage 3: GRPO Training
# ============================================================
echo ""
echo "######################################"
echo "# Stage 3: GRPO Training             #"
echo "######################################"
echo ""

bash "${SCRIPT_DIR}/stage3_grpo_train.sh" "${MERGED_OUTPUT}" "${MODEL_TYPE}" "${DATASET}" "${SEED}" "${GPU}"

# Find best GRPO checkpoint
# ms-swift creates: output_dir/v0-YYYYMMDD-HHMMSS/checkpoint-*
GRPO_VERSION_DIR=$(ls -d ${GRPO_OUTPUT}/v*-* 2>/dev/null | sort -V | tail -1)
if [ -z "$GRPO_VERSION_DIR" ]; then
    echo "ERROR: No GRPO version directory found in ${GRPO_OUTPUT}"
    exit 1
fi
GRPO_CHECKPOINT=$(ls -d ${GRPO_VERSION_DIR}/checkpoint-* 2>/dev/null | sort -V | tail -1)
if [ -z "$GRPO_CHECKPOINT" ]; then
    echo "ERROR: No GRPO checkpoint found in ${GRPO_VERSION_DIR}"
    exit 1
fi

# ============================================================
# Stage 4: Inference & Evaluation
# ============================================================
echo ""
echo "######################################"
echo "# Stage 4: Inference & Evaluation    #"
echo "######################################"
echo ""

bash "${SCRIPT_DIR}/inference.sh" "${GRPO_CHECKPOINT}" "${DATASET}" "${MODEL_NAME}_${DATASET}_seed${SEED}_results" "${GPU}"

# ============================================================
# Summary
# ============================================================
echo ""
echo "=============================================="
echo "Pipeline Completed Successfully!"
echo "=============================================="
echo ""
echo "Outputs:"
echo "  Stage 1 (SFT):    ${SFT_OUTPUT}"
echo "  Stage 2 (Merged): ${MERGED_OUTPUT}"
echo "  Stage 3 (GRPO):   ${GRPO_OUTPUT}"
echo "  Inference:        ${OUTPUT_DIR}/inference/"
echo ""
echo "Evaluation results: ${OUTPUT_DIR}/inference/eval_results.json"
echo "=============================================="
