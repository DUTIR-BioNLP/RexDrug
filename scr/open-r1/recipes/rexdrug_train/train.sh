CUDA_VISIBLE_DEVICES=0 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
    --main_process_port 10000 \
    --num_processes=1 src/open_r1/grpo_for_rexdrug.py \
    --config submit/RexDrug/RL/recipes/rexdrug_train/config.yaml \
    