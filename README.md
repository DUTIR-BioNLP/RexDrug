# RexDrug: Reliable Multi-Drug Combination Extraction through Reasoning-Enhanced LLMs

## Overview
Automated Drug Combination Extraction (DCE) from large-scale biomedical literature is important for precision medicine and pharmacological research. However, existing extraction methodologies predominantly focus on binary interactions. When tasked with complex, variable-length n-ary pharmacological relationships, these methods often struggle to capture the underlying compatibility logic between multiple agents and lack interpretability. To address these limitations, we introduce RexDrug, an end-to-end n-ary drug combination extraction framework that leverages Reinforcement Learning (RL) to enhance the reasoning capabilities of Large Language Models (LLMs). Within this framework, a two-stage training strategy is implemented: first, a multi-agent collaborative system is utilized to automatically generate reasoning traces that emulate expert logic, forming the basis for Supervised Fine-Tuning (SFT). Subsequently, we employ a multi-dimensional reward function specifically tailored for drug combination identification to further refine reasoning quality and extraction precision. Experimental results show that RexDrug outperforms current state-of-the-art baselines in extraction accuracy, with strong robustness. Human expert evaluation further indicates that RexDrug produces coherent medical reasoning while correctly identifying complex therapeutic regimens. These findings demonstrate the reliability of RexDrug as a scalable and interpretable tool for large-scale biomedical literature mining

[](/RexDrug/pic/zongtu.png)

## Project Structure

```
RexDrug/
├── README.md                    # This file
├── setup.py                     # Installation script
├── requirements.txt             # Python dependencies
│
├── datasets/                    # Datasets and data processing
│   ├── DDI13/                   # DDI2013 dataset
│   │   ├── DDI2013/             # Raw data (train.jsonl, test.jsonl)
│   │   ├── cot_data_get/        # CoT generation & evaluation scripts
│   │   ├── sft_cot_data/        # SFT format with CoT (generated)
│   │   ├── grpo_cot_data/       # GRPO format with CoT (generated)
│   │   └── sft_only_re/         # SFT format without CoT
│   └── DrugComb/                # DrugComb dataset
│       ├── ord_data/            # Raw data (train.jsonl, test.jsonl)
│       ├── cot_data_get/        # CoT generation & evaluation scripts
│       ├── sft_cot_data/        # SFT format with CoT (included)
│       ├── grpo_cot_data_v2/    # GRPO format with CoT (included)
│       └── sft_only_re/         # SFT format without CoT
│
├── swift/                       # ms-swift framework core
│   └── rewards/
│       └── grpo_reward.py       # Custom GRPO reward functions
│
├── scripts/                     # Training and inference scripts
│   ├── config.sh                # Configuration file
│   ├── stage1_sft_train.sh      # Stage 1: SFT training
│   ├── stage2_merge_lora.sh     # Stage 2: Merge LoRA
│   ├── stage3_grpo_train.sh     # Stage 3: GRPO training
│   ├── inference.sh             # Inference script
│   └── run_full_pipeline.sh     # Full pipeline (all stages)
│
├── eval/                        # Evaluation scripts
│   ├── eval_files_drugcomb.py   # DrugComb evaluation
│   └── eval_files_ddi.py        # DDI13 evaluation
│
└── outputs/                     # Training outputs (generated)
```

## Requirements

- Python 3.10.0
- PyTorch 2.8.0
- CUDA 12.8

## Installation

### 1. Create a virtual environment

```bash
conda create -n drugcomb python=3.10
conda activate drugcomb
```

### 2. Install dependencies

```bash
cd RexDrug
pip install -e .
pip install vllm==0.11.0
```

### 3. Configure model paths

Edit `scripts/config.sh` and set the paths to your local models:

```bash
# LLaMA 3.1-8B-Instruct model path
export LLAMA_MODEL_PATH="/path/to/llama3.1-8b-instruct"

# Qwen 2.5-7B-Instruct model path
export QWEN_MODEL_PATH="/path/to/Qwen2.5-7B-Instruct"
```

## Data Preparation

### DrugComb Dataset

The DrugComb CoT data is **pre-generated and included** in:
- `datasets/DrugComb/sft_cot_data/` (SFT format, 1098 training + 272 test samples)
- `datasets/DrugComb/grpo_cot_data_v2/` (GRPO format, 1098 training + 272 test samples)

No additional data preparation is needed for DrugComb.

To regenerate CoT data from scratch (optional):

```bash
# Step 1: Generate CoT reasoning chains
#   --gen-model: model for CoT generation (default: gpt-4o)
#   --eval-model: model for CoT quality evaluation (default: gpt-5.1)
python datasets/DrugComb/cot_data_get/get_cot_data_for_rel_ext.py \
    --input datasets/DrugComb/ord_data/train.jsonl \
    --output datasets/DrugComb/cot_data_get/output \
    --api-key YOUR_API_KEY \
    --gen-model gpt-4o \
    --eval-model gpt-5.1 \
    --threads 50

# Step 2: Convert to SFT format
python datasets/DrugComb/sft_cot_data/convert_cot_to_sft.py \
    --input datasets/DrugComb/cot_data_get/output/train_final_cot.jsonl \
    --output datasets/DrugComb/sft_cot_data/drugcomb_train.jsonl

# Step 3: Convert to GRPO format
python datasets/DrugComb/grpo_cot_data_v2/convert_cot_to_grpo.py \
    --input datasets/DrugComb/cot_data_get/output/train_final_cot.jsonl \
    --output datasets/DrugComb/grpo_cot_data_v2/train_drugcomb.jsonl
```

### DDI13 Dataset

The DDI13 raw data is included in `datasets/DDI13/DDI2013/`. CoT data must be generated before training:

```bash
# Step 1: Generate CoT reasoning chains
#   --gen-model: model for CoT generation (default: gpt-4o)
#   --eval-model: model for CoT quality evaluation (default: gpt-5.1)
python datasets/DDI13/cot_data_get/get_cot_data.py \
    --input datasets/DDI13/DDI2013/train.jsonl \
    --output datasets/DDI13/cot_data_get/output \
    --api-key YOUR_API_KEY \
    --gen-model gpt-4o \
    --eval-model gpt-5.1 \
    --threads 10

# Step 2: Convert to SFT format
python datasets/DDI13/cot_data_get/convert_cot_to_sft.py \
    --input datasets/DDI13/cot_data_get/output/final_cot_YYYYMMDD_HHMMSS.jsonl \
    --output datasets/DDI13/sft_cot_data/train_ddi.jsonl

# Step 3: Convert to GRPO format
python datasets/DDI13/cot_data_get/convert_cot_to_grpo.py \
    --input datasets/DDI13/cot_data_get/output/final_cot_YYYYMMDD_HHMMSS.jsonl \
    --output datasets/DDI13/grpo_cot_data/train_ddi.jsonl
```

## Usage

### Run the Full Pipeline

The easiest way to run the complete training pipeline:

```bash
cd RexDrug

# Run with LLaMA on DrugComb dataset
bash scripts/run_full_pipeline.sh llama drugcomb 42 0

# Run with Qwen on DDI13 dataset
bash scripts/run_full_pipeline.sh qwen ddi 42 0
```

**Arguments:**
- `model_type`: `llama` or `qwen`
- `dataset`: `drugcomb` or `ddi`
- `seed`: Random seed (default: 42)
- `gpu`: GPU device ID (default: 0)

### Run Individual Stages

#### Stage 1: SFT Training

```bash
bash scripts/stage1_sft_train.sh llama drugcomb 42 0
```

#### Stage 2: Merge LoRA

Note: ms-swift creates output directories with format `v0-YYYYMMDD-HHMMSS/checkpoint-xxx`.

```bash
# Find the checkpoint path first
ls outputs/stage1_sft/llama3_1_sft_cot_drugcomb_seed42/v0-*/checkpoint-*

# Then run merge
bash scripts/stage2_merge_lora.sh outputs/stage1_sft/llama3_1_sft_cot_drugcomb_seed42/v0-YYYYMMDD-HHMMSS/checkpoint-xxx
```

#### Stage 3: GRPO Training

```bash
bash scripts/stage3_grpo_train.sh outputs/stage2_merged/llama3_1_drugcomb_seed42 llama drugcomb 42 0
```

#### Inference

```bash
# Find the GRPO checkpoint
ls outputs/stage3_grpo/llama3_1_grpo_cot_drugcomb_seed42/v0-*/checkpoint-*

# Run inference
bash scripts/inference.sh outputs/stage3_grpo/llama3_1_grpo_cot_drugcomb_seed42/v0-YYYYMMDD-HHMMSS/checkpoint-xxx drugcomb results 0
```

## GRPO Reward Functions

Custom reward functions are defined in `swift/rewards/grpo_reward.py`. The reward functions guide the GRPO training process:

| Reward Function | Description |
|-----------------|-------------|
| `*_cot_format` | Validates `<think>` and `<answer>` tag structure |
| `*_cot_think` | Checks reasoning quality in think section |
| `*_coverage_cot` | Measures entity coverage (Jaccard similarity) |
| `*_accuracy_cot` | F1-based accuracy reward |

## Configuration

Key hyperparameters can be modified in `scripts/config.sh`:

| Parameter | SFT Stage | GRPO Stage |
|-----------|-----------|------------|
| LoRA Rank | 16 | 16 |
| LoRA Alpha | 32 | 32 |
| Learning Rate | 1e-5 | 1e-6 |
| Epochs | 10 | 20 |
| Batch Size | 1 | 4 |

### Note on Template Names

In the training scripts, the `--template` parameter is set to `llama3_2` for LLaMA 3.1-8B-Instruct. This is a naming convention defined by the ms-swift framework where `llama3_2` refers to the chat template format used by the LLaMA 3.x Instruct model family (including 3.1). It does not imply that a LLaMA 3.2 model is used. Similarly, `qwen2_5` is the ms-swift template name for Qwen 2.5 models.

## Troubleshooting

### CUDA Out of Memory

- Reduce `vllm_gpu_memory_utilization` in scripts
- Add `--gradient_checkpointing true`
- Reduce batch size

### Model Path Not Found

Ensure base models are downloaded and paths are correctly set in `scripts/config.sh`.

## License

Apache License 2.0

## Acknowledgments

Built upon the [ms-swift](https://github.com/modelscope/swift) framework by ModelScope.
