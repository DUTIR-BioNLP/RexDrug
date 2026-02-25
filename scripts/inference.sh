#!/bin/bash
# ============================================================
# Inference Script
#
# This script runs inference on test data using trained model.
# ============================================================

set -e

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# Parse arguments
CHECKPOINT="${1}"
DATASET="${2:-drugcomb}"       # drugcomb or ddi
OUTPUT_NAME="${3:-inference_results}"
GPU="${4:-0}"

if [ -z "$CHECKPOINT" ]; then
    echo "Usage: $0 <checkpoint_path> <dataset> [output_name] [gpu]"
    echo ""
    echo "Arguments:"
    echo "  checkpoint_path: Path to the trained model checkpoint"
    echo "  dataset: drugcomb or ddi"
    echo "  output_name: Output file name (default: inference_results)"
    echo "  gpu: GPU device ID (default: 0)"
    echo ""
    echo "Example:"
    echo "  $0 outputs/stage3_grpo/llama3_1_grpo_cot_drugcomb_seed42/v0-YYYYMMDD-HHMMSS/checkpoint-1000 drugcomb"
    exit 1
fi

# Validate checkpoint
if [ ! -d "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

# Set test dataset
if [ "$DATASET" == "drugcomb" ]; then
    TEST_DATA="${DRUGCOMB_DIR}/grpo_cot_data_v2/test_drugcomb.jsonl"
else
    TEST_DATA="${DDI13_DIR}/grpo_cot_data/test_ddi.jsonl"
fi

# Output file
OUTPUT_FILE="${OUTPUT_DIR}/inference/${OUTPUT_NAME}.jsonl"
mkdir -p "$(dirname "${OUTPUT_FILE}")"

echo "=== Inference ==="
echo "Checkpoint: ${CHECKPOINT}"
echo "Test Data:  ${TEST_DATA}"
echo "Output:     ${OUTPUT_FILE}"
echo "GPU:        ${GPU}"
echo "================="

# Validate test data
if [ ! -f "${TEST_DATA}" ]; then
    echo "ERROR: Test data not found: ${TEST_DATA}"
    exit 1
fi

# Run inference
CUDA_VISIBLE_DEVICES=${GPU} swift infer \
    --adapters "${CHECKPOINT}" \
    --infer_backend vllm \
    --vllm_gpu_memory_utilization 0.85 \
    --vllm_max_model_len 4096 \
    --temperature 0.1 \
    --max_new_tokens 1024 \
    --val_dataset "${TEST_DATA}" \
    --result_path "${OUTPUT_FILE}"

echo ""
echo "=== Inference Completed ==="
echo "Results saved to: ${OUTPUT_FILE}"
echo ""
echo "Running evaluation..."

# Run evaluation
if [ "$DATASET" == "drugcomb" ]; then
    python "${EVAL_DIR}/eval_files_drugcomb.py" \
        --data-dir "$(dirname "${OUTPUT_FILE}")" \
        --output "$(dirname "${OUTPUT_FILE}")/eval_results.json" \
        --mode flat
else
    python "${EVAL_DIR}/eval_files_ddi.py" \
        --data-dir "$(dirname "${OUTPUT_FILE}")" \
        --output "$(dirname "${OUTPUT_FILE}")/eval_results.json" \
        --mode flat
fi

echo ""
echo "=== Evaluation Completed ==="
echo "Results saved to: $(dirname "${OUTPUT_FILE}")/eval_results.json"
