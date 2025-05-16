#!/bin/bash

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
        out_dir="small_experiments/scaled_loop1" \
        wandb_run_name="jz-4E-4Lx3-4D" \
        model.n_encoder=4 \
        model.n_layer=4 \
        model.n_loop=3 \
        model.n_decoder=4 \
        model.n_head=12 \
        model.n_embd=768 \
        model.use_loop_weight=False \
        init_from='scratch' \
        compile=True \
        batch_size=32 \
        learning_rate=6e-4 \
        dataset=fineweb-edu \
        max_iters=2_000 \
        lr_decay_iters=2_000 \
        gradient_accumulation_steps=128 \
        eval_interval=10 \
        eval_iters=50 \
        warmup_iters=100 \
        always_save_checkpoint=False

torchrun --nproc_per_node=gpu train.py \
        out_dir="small_experiments/scaled_loop2" \
        wandb_run_name="jz-2E-4Lx3-6D" \
        model.n_encoder=2 \
        model.n_layer=4 \
        model.n_loop=3 \
        model.n_decoder=6 \
        model.n_head=12 \
        model.n_embd=768 \
        model.use_loop_weight=False \
        init_from='scratch' \
        compile=True \
        batch_size=32 \
        learning_rate=6e-4 \
        dataset=fineweb-edu \
        max_iters=2_000 \
        lr_decay_iters=2_000 \
        gradient_accumulation_steps=128 \
        eval_interval=10 \
        eval_iters=50 \
        warmup_iters=100 \
        always_save_checkpoint=True


# time taken
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Total time taken: $((duration / 60)) minutes and $((duration % 60)) seconds"