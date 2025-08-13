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
export NCCL_P2P_LEVEL=PXB # only on non nvlink

torchrun --nproc_per_node=gpu train.py \
        out_dir="small_experiments/test" \
        wandb_run_name="nv-a100-speedtest-fsdp" \
        model.n_encoder=0 \
        model.n_layer=12 \
        model.n_loop=1 \
        model.n_decoder=0 \
        model.n_head=12 \
        model.n_embd=768 \
        model.use_loop_weight=False \
        init_from='scratch' \
        batch_size=32 \
        fsdp=True \
        learning_rate=6e-4 \
        max_iters=20 \
        wandb_log=False \
        gradient_accumulation_steps=128 \
        dataset=fineweb-edu \
        eval_interval=20 \
        eval_iters=50 \
        warmup_iters=100 \
        always_save_checkpoint=False compile=True

# time taken
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Total time taken: $((duration / 60)) minutes and $((duration % 60)) seconds"