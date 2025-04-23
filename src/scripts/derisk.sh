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

torchrun --nproc_per_node=$N_GPUS train.py \
        out_dir="small_experiments/test" \
        wandb_run_name="derisk" \
        model.n_encoder=0 \
        model.n_layer=4 \
        model.n_loop=3 \
        model.n_head=12 \
        model.n_embd=768 \
        init_from='scratch' \
        model.dropout=0.1 \
        batch_size=8 \
        learning_rate=6e-4 \
        max_iters=20 \
        wandb_log=False \
        gradient_accumulation_steps=128 \
        compile=True \
        dataset=fineweb-edu
        # debug=True


# time taken
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Total time taken: $((duration / 60)) minutes and $((duration % 60)) seconds"