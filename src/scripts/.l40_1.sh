#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status


N_GPUS=$(nvidia-smi -L | wc -l)
if [ $N_GPUS -lt 1 ]; then
    echo "No GPUs found. Exiting."
    exit 1
fi

mkdir -p small_experiments

start_time=$(date +%s)


export NCCL_P2P_LEVEL=PXB

torchrun --nproc_per_node=gpu train.py \
        out_dir="small_experiments/2E-4Lx4-0D-w" \
        wandb_run_name="residual-2E-4Lx4-0D" \
        model.n_encoder=2 \
        model.n_layer=4 \
        model.n_loop=4 \
        model.n_decoder=0 \
        model.n_head=4 \
        model.n_embd=256 \
        model.use_loop_weight=True \
        init_from='scratch' \
        batch_size=32 \
        learning_rate=6e-4 \
        dataset=fineweb-edu \
        max_iters=2_000 \
        lr_decay_iters=2_000 \
        gradient_accumulation_steps=128 \
        eval_interval=10 \
        eval_iters=50 \
        warmup_iters=100 \
        always_save_checkpoint=False \
        wandb_log=True

torchrun --nproc_per_node=gpu train.py \
        out_dir="small_experiments/1E-4Lx4-1D-w" \
        wandb_run_name="residual-1E-4Lx4-1D" \
        model.n_encoder=1 \
        model.n_layer=4 \
        model.n_loop=4 \
        model.n_decoder=1 \
        model.n_head=4 \
        model.n_embd=256 \
        model.use_loop_weight=True \
        init_from='scratch' \
        batch_size=32 \
        learning_rate=6e-4 \
        dataset=fineweb-edu \
        max_iters=2_000 \
        lr_decay_iters=2_000 \
        gradient_accumulation_steps=128 \
        eval_interval=10 \
        eval_iters=50 \
        warmup_iters=100 \
        always_save_checkpoint=False \
        wandb_log=True

torchrun --nproc_per_node=gpu train.py \
        out_dir="small_experiments/0E-4Lx4-2D-w" \
        wandb_run_name="residual-0E-4Lx4-2D" \
        model.n_encoder=0 \
        model.n_layer=4 \
        model.n_loop=4 \
        model.n_decoder=2 \
        model.n_head=4 \
        model.n_embd=256 \
        model.use_loop_weight=True \
        init_from='scratch' \
        batch_size=32 \
        learning_rate=6e-4 \
        dataset=fineweb-edu \
        max_iters=2_000 \
        lr_decay_iters=2_000 \
        gradient_accumulation_steps=128 \
        eval_interval=10 \
        eval_iters=50 \
        warmup_iters=100 \
        always_save_checkpoint=False \
        wandb_log=True

# time taken
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Total time taken: $((duration / 60)) minutes and $((duration % 60)) seconds"