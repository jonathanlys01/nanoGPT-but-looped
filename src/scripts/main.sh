#!/bin/bash
# Bash script to run smaller-scale Loop-Residual Neural Network experiments
# This script focuses on smaller models for faster experimentation

set -e  # Exit immediately if a command exits with a non-zero status

# Common configuration parameters - using smaller values for quicker experiments
BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=64  # Adjusted for smaller batch size
LEARNING_RATE=6e-4
DATASET="openwebtext"
WANDB_LOG=True
MAX_ITERS=40_000
DROPOUT=0.1      # Adding some dropout for these smaller models

N_GPUS=$(nvidia-smi -L | wc -l)
if [ $N_GPUS -lt 1 ]; then
    echo "No GPUs found. Exiting."
    exit 1
fi

mkdir -p small_experiments

start_time=$(date +%s)

COMMON="model.n_head=4 \
        model.n_embd=256 \
        init_from='scratch' \
        model.dropout=$DROPOUT \
        batch_size=$BATCH_SIZE \
        gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
        learning_rate=$LEARNING_RATE \
        dataset=$DATASET \
        max_iters=$MAX_ITERS \
        lr_decay_iters=$MAX_ITERS \
        wandb_log=$WANDB_LOG
        compile=True"

export OMP_NUM_THREADS=2

# Exp 1 (2 E / 4 L x 4 Loop / 0 D)
torchrun --nproc_per_node=gpu train.py \
        out_dir="small_experiments/exp1" \
        wandb_run_name="long-2E-4Lx4-0D" \
        model.n_encoder=2 \
        model.n_layer=4 \
        model.n_loop=4 \
        model.n_decoder=0 \
        $COMMON

# Exp 2 (0 E / 6 L x 3 Loop / 0 D)
torchrun --nproc_per_node=gpu train.py \
        out_dir="small_experiments/exp2" \
        wandb_run_name="long-0E-6Lx3-0D" \
        model.n_encoder=0 \
        model.n_layer=6 \
        model.n_loop=3 \
        model.n_decoder=0 \
        $COMMON

# Exp 3 (1 E / 4 L x 4 Loop / 1 D)
torchrun --nproc_per_node=gpu train.py \
        out_dir="small_experiments/exp3" \
        wandb_run_name="long-1E-4Lx4-1D" \
        model.n_encoder=1 \
        model.n_layer=4 \
        model.n_loop=4 \
        model.n_decoder=1 \
        $COMMON

# time taken
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Total time taken: $((duration / 60)) minutes and $((duration % 60)) seconds"