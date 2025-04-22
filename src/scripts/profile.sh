#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status


N_GPUS=$(nvidia-smi -L | wc -l)
if [ $N_GPUS -lt 1 ]; then
    echo "No GPUs found. Exiting."
    exit 1
fi

echo "Number of GPUs available: $N_GPUS"

start_time=$(date +%s)

export OMP_NUM_THREADS=2

torchrun --nproc_per_node=gpu _profile.py \
        out_dir="small_experiments/test" \
        wandb_run_name="derisk" \
        model.n_encoder=0 \
        model.n_layer=4 \
        model.n_loop=2 \
        model.n_head=4 \
        model.n_embd=256 \
        init_from='scratch' \
        model.dropout=0.1 \
        batch_size=8 \
        learning_rate=6e-4 \
        dataset=openwebtext \
        max_iters=2 \
        wandb_log=False \
        dtype=float16 \
        compile=True \
        profile=True
        # debug=True


# time taken
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Total time taken: $((duration / 60)) minutes and $((duration % 60)) seconds"