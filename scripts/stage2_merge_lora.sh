#!/bin/bash
# ============================================================
# Stage 2: Merge LoRA Weights
#
# This script merges the LoRA adapter weights from Stage 1
# into the base model, creating a merged model for Stage 3.
# ============================================================

set -e

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# Parse arguments
SFT_CHECKPOINT="${1}"
OUTPUT_NAME="${2:-merged_model}"

if [ -z "$SFT_CHECKPOINT" ]; then
    echo "Usage: $0 <sft_checkpoint_path> [output_name]"
    echo ""
    echo "Arguments:"
    echo "  sft_checkpoint_path: Path to the SFT checkpoint directory"
    echo "  output_name: Name for the merged model (default: merged_model)"
    echo ""
    echo "Example:"
    echo "  $0 outputs/stage1_sft/llama3_1_sft_cot_drugcomb_seed42/checkpoint-1000"
    exit 1
fi

# Validate checkpoint
if [ ! -d "$SFT_CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $SFT_CHECKPOINT"
    exit 1
fi

# Output directory
MERGED_OUTPUT="${OUTPUT_DIR}/stage2_merged/${OUTPUT_NAME}"

echo "=== Stage 2: Merge LoRA Weights ==="
echo "SFT Checkpoint: ${SFT_CHECKPOINT}"
echo "Merged Output:  ${MERGED_OUTPUT}"
echo "==================================="

# Run merge
swift export \
    --adapters "${SFT_CHECKPOINT}" \
    --merge_lora true \
    --output_dir "${MERGED_OUTPUT}"

echo ""
echo "=== Stage 2 Completed ==="
echo "Merged model saved to: ${MERGED_OUTPUT}"
echo ""
echo "Next step: Run stage3_grpo_train.sh to perform GRPO training"
