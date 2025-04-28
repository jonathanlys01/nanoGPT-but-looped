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
        out_dir="small_experiments/test" \
        wandb_run_name="derisk" \
        model.n_encoder=0 \
        model.n_layer=12 \
        model.n_loop=1 \
        model.n_head=12 \
        model.n_embd=768 \
        init_from='scratch' \
        batch_size=32 \
        learning_rate=6e-4 \
        max_iters=100 \
        wandb_log=False \
        gradient_accumulation_steps=128 \
        compile=True \
        dataset=fineweb-edu \
        eval_interval=10 \
        eval_iters=50 \
        warmup_iters=100 \
        always_save_checkpoint=True
        # less eval_iters bc parallelized across GPUs
        # debug=True


# time taken
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Total time taken: $((duration / 60)) minutes and $((duration % 60)) seconds"