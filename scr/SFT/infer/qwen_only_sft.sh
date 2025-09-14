
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# 基础参数
MODEL_PATH="/data/SharedFile/Qwen/Qwen2___5-7B-Instruct"
DATASET="drugcomb_sft_method_3_new_output_test"
DATASET_DIR="data"
TEMPLATE="qwen"
CHECKPOINTS=()
for CKPT in "${CHECKPOINTS[@]}"; do
    ADAPTER_PATH="checkpoint-${CKPT}"
    SAVE_NAME="only_sft/${CKPT}.jsonl"

    echo ">>> Running inference for checkpoint ${CKPT}"
    
    python scripts/vllm_infer.py \
        --model_name_or_path "${MODEL_PATH}" \
        --adapter_name_or_path "${ADAPTER_PATH}" \
        --dataset "${DATASET}" \
        --dataset_dir "${DATASET_DIR}" \
        --template "${TEMPLATE}" \
        --save_name "${SAVE_NAME}"
done