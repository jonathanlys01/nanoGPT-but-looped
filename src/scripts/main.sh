#!/bin/bash
# Bash script to run smaller-scale Loop-Residual Neural Network experiments
# This script focuses on smaller models for faster experimentation

set -e  # Exit immediately if a command exits with a non-zero status

N_GPUS=$(nvidia-smi -L | wc -l)
if [ $N_GPUS -lt 1 ]; then
    echo "No GPUs found. Exiting."
    exit 1
fi

mkdir -p small_experiments

start_time=$(date +%s)



export OMP_NUM_THREADS=2

torchrun --nproc_per_node=gpu train.py \
        out_dir="small_experiments/gpt2" \
        wandb_run_name="gpt2-124M-true_skip" \
        model.n_encoder=0 \
        model.n_layer=12 \
        model.n_loop=1 \
        model.n_decoder=0 \
        model.n_head=12 \
        model.n_embd=768 \
        model.use_loop_weight=False \
        model.skip_first=True \
        init_from='scratch' \
        compile=True \
        batch_size=32 \
        learning_rate=6e-4 \
        dataset=fineweb-edu \
        max_iters=2_000 \
        lr_decay_iters=2_000 \
        gradient_accumulation_steps=128 \
        eval_interval=20 \
        eval_iters=50 \
        warmup_iters=100 \
        always_save_checkpoint=False

# time taken
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Total time taken: $((duration / 60)) minutes and $((duration % 60)) seconds"