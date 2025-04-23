#!/bin/bash

#SBATCH --account=vaz@a100
#SBATCH --constraint=a100

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread

#SBATCH --job-name=looped-nanogpt
#SBATCH --output=slurm-logs/%j-%x.out
#SBATCH --error=slurm-logs/%j-%x.err
#SBATCH --time=00:30:00
#SBATCH --qos=qos_gpu_a100-dev
# Load the modules


echo "Job started at $(date)"
export WANDB_MODE=offline

module purge
module load arch/a100
module load pytorch-gpu/py3/2.6.0

mkdir -p slurm-logs
mkdir -p experiments

set -x
srun python train.py \
        out_dir="experiments/baseline" \
        wandb_run_name="baseline" \
        wandb_log=True \
        model.n_encoder=0 \
        model.n_layer=12 \
        model.n_loop=1 \
        model.n_head=12 \
        model.n_embd=768 \
        init_from='scratch' \
        batch_size=32 \
        learning_rate=6e-4 \
        dataset=fineweb-edu \
        max_iters=100_000 \
        lr_decay_iters=100_000 \
        gradient_accumulation_steps=128