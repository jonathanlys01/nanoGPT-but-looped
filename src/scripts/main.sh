#!/bin/bash
# Bash script to run smaller-scale Loop-Residual Neural Network experiments
# This script focuses on smaller models for faster experimentation

set -e  # Exit immediately if a command exits with a non-zero status

# Common configuration parameters - using smaller values for quicker experiments
BATCH_SIZE=8
LEARNING_RATE=6e-4
DATASET="openwebtext"
WANDB_LOG=True
MAX_ITERS=10_000  # Reduced significantly for faster experimentation
DROPOUT=0.1      # Adding some dropout for these smaller models

N_GPUS=$(lspci | grep -i 'nvidia' | wc -l)

mkdir -p small_experiments

start_time=$(date +%s)

COMMON="model.n_head=4 \
        model.n_embd=256 \
        init_from='scratch' \
        model.dropout=$DROPOUT \
        batch_size=$BATCH_SIZE \
        learning_rate=$LEARNING_RATE \
        dataset=$DATASET \
        max_iters=$MAX_ITERS \
        wandb_log=$WANDB_LOG"


# Exp3 (0 E / 4 L / 2 Loop)
torchrun --nproc_per_node=$N_GPUS train.py \
        out_dir="small_experiments/exp3" \
        wandb_run_name="0E-4L-2Loop(no-enc)" \
        model.n_encoder=0 \
        model.n_layer=4 \
        model.n_loop=2 \
        $COMMON

##############################################################################

# # Baseline (0 E / 4 L / 1 Loop)
# torchrun --nproc_per_node=$N_GPUS train.py \
#         out_dir="small_experiments/baseline" \
#         wandb_run_name="0E-4L-1Loop(baseline)" \
#         model.n_encoder=0 \
#         model.n_layer=4 \
#         model.n_loop=1 \
#         $COMMON

# # Sanity check (1 E / 3 L / 1 Loop) (should be similar to baseline)
# torchrun --nproc_per_node=$N_GPUS train.py \
#         out_dir="small_experiments/sanity_check" \
#         wandb_run_name="1E-3L-1Loop(sanity_check)" \
#         model.n_encoder=1 \
#         model.n_layer=3 \
#         model.n_loop=1 \
#         $COMMON

# # Exp 1 (0 E / 2 L / 2 Loop)
# torchrun --nproc_per_node=$N_GPUS train.py \
#         out_dir="small_experiments/exp1" \
#         wandb_run_name="0E-2L-2Loop(isoflop)" \
#         model.n_encoder=0 \
#         model.n_layer=2 \
#         model.n_loop=2 \
#         $COMMON

# # Exp 2 (1 E / 3 L / 2 Loop)
# torchrun --nproc_per_node=$N_GPUS train.py \
#         out_dir="small_experiments/exp2" \
#         wandb_run_name="1E-3L-2Loop(isoparam)" \
#         model.n_encoder=1 \
#         model.n_layer=3 \
#         model.n_loop=2 \
#         $COMMON

# # Exp 3 (1 E / 3 L / 4 Loop)
# torchrun --nproc_per_node=$N_GPUS train.py \
#         out_dir="small_experiments/exp3" \
#         wandb_run_name="1E-3L-4Loop(isoparammax)" \
#         model.n_encoder=1 \
#         model.n_layer=3 \
#         model.n_loop=4 \
#         $COMMON

# time taken
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Total time taken: $((duration / 60)) minutes and $((duration % 60)) seconds"