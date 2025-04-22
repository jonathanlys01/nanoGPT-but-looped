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
MAX_ITERS=10_000  # Reduced significantly for faster experimentation
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
        wandb_log=$WANDB_LOG
        compile=True"



# # Baseline (0 E / 4 L / 1 Loop)
# torchrun --nproc_per_node=$N_GPUS train.py \
#         out_dir="small_experiments/baseline" \
#         wandb_run_name="re-0E-4L-1Loop" \
#         model.n_encoder=0 \
#         model.n_layer=4 \
#         model.n_loop=1 \
#         $COMMON


# # Exp 1 (1 E / 3 L / 2 Loop)
# torchrun --nproc_per_node=$N_GPUS train.py \
#         out_dir="small_experiments/exp2" \
#         wandb_run_name="re-1E-3L-2Loop" \
#         model.n_encoder=1 \
#         model.n_layer=3 \
#         model.n_loop=2 \
#         $COMMON

# # Exp 2 (1 E / 3 L / 4 Loop)
# torchrun --nproc_per_node=$N_GPUS train.py \
#         out_dir="small_experiments/exp2" \
#         wandb_run_name="re-1E-3L-4Loop" \
#         model.n_encoder=1 \
#         model.n_layer=3 \
#         model.n_loop=4 \
#         $COMMON

# # Exp 3 (1 E / 3 L / 8 Loop)
# torchrun --nproc_per_node=$N_GPUS train.py \
#         out_dir="small_experiments/exp3" \
#         wandb_run_name="re-1E-3L-8Loop" \
#         model.n_encoder=1 \
#         model.n_layer=3 \
#         model.n_loop=8 \
#         $COMMON

# Exp 4 (1 E / 3 L / 4 Loop, loop pe)
torchrun --nproc_per_node=$N_GPUS train.py \
        out_dir="small_experiments/exp4" \
        wandb_run_name="re-1E-3L-4Loop-pe" \
        model.n_encoder=1 \
        model.n_layer=3 \
        model.n_loop=4 \
        model.use_loop_pe=True \
        $COMMON

# time taken
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Total time taken: $((duration / 60)) minutes and $((duration % 60)) seconds"