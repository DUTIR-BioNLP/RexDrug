#!/bin/bash
# ============================================================
# Two-Stage Drug Combination Extraction with GRPO
# Project Configuration
# ============================================================

# Get project root directory (auto-detected)
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Dataset paths (relative to project root)
export DATASETS_DIR="${PROJECT_ROOT}/datasets"
export DDI13_DIR="${DATASETS_DIR}/DDI13"
export DRUGCOMB_DIR="${DATASETS_DIR}/DrugComb"

# GRPO reward plugin path
export PLUGIN_PATH="${PROJECT_ROOT}/swift/rewards/grpo_reward.py"

# Output directory
export OUTPUT_DIR="${PROJECT_ROOT}/outputs"
mkdir -p "${OUTPUT_DIR}"

# Evaluation scripts
export EVAL_DIR="${PROJECT_ROOT}/eval"

# ============================================================
# MODEL PATHS - PLEASE MODIFY ACCORDING TO YOUR ENVIRONMENT
# ============================================================
# LLaMA 3.1-8B-Instruct model path
export LLAMA_MODEL_PATH="/path/to/llama3.1-8b-instruct"


# Qwen 2.5-7B-Instruct model path
export QWEN_MODEL_PATH="/path/to/Qwen2.5-7B-Instruct"
# ============================================================

# Training hyperparameters
export DEFAULT_LORA_RANK=16
export DEFAULT_LORA_ALPHA=32
export DEFAULT_SFT_LR="1e-5"
export DEFAULT_GRPO_LR="1e-6"
export DEFAULT_SFT_EPOCHS=10
export DEFAULT_GRPO_EPOCHS=20
export DEFAULT_SFT_BATCH_SIZE=1
export DEFAULT_GRPO_BATCH_SIZE=4

# Multi-seed experiments
export SEEDS=(42 123 2025 3407 6666)

# Print configuration
print_config() {
    echo "=============================================="
    echo "Project Configuration"
    echo "=============================================="
    echo "PROJECT_ROOT: ${PROJECT_ROOT}"
    echo "DATASETS_DIR: ${DATASETS_DIR}"
    echo "PLUGIN_PATH:  ${PLUGIN_PATH}"
    echo "OUTPUT_DIR:   ${OUTPUT_DIR}"
    echo "LLAMA_MODEL:  ${LLAMA_MODEL_PATH}"
    echo "QWEN_MODEL:   ${QWEN_MODEL_PATH}"
    echo "=============================================="
}

# Validate paths
validate_paths() {
    local has_error=0

    if [ ! -d "${DATASETS_DIR}" ]; then
        echo "ERROR: Datasets directory not found: ${DATASETS_DIR}"
        has_error=1
    fi

    if [ ! -f "${PLUGIN_PATH}" ]; then
        echo "ERROR: Plugin file not found: ${PLUGIN_PATH}"
        has_error=1
    fi

    return $has_error
}
