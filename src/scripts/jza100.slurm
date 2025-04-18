#!/bin/bash

#SBATCH --account=vaz@a100
#SBATCH --constraint=a100

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread

#SBATCH --job-name=looped-nanogpt
#SBATCH --output=slurm-logs/%j-%x-%j.out
#SBATCH --error=slurm-logs/%j-%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --qos=qos_gpu_a100-dev
# Load the modules

set -x

echo "Job started at $(date)"
export WANDB_MODE=offline

module purge
module load arch/a100
module load pytorch-gpu/py3/2.6.0

mkdir -p slurm-logs
mkdir -p small_experiments

srun python train.py \
        out_dir="small_experiments/test" \
        wandb_run_name="derisk_run" \
        model.n_encoder=0 \
        model.n_layer=4 \
        model.n_loop=2 \
        model.n_head=4 \
        model.n_embd=256 \
        init_from='scratch' \
        model.dropout=0.1 \
        batch_size=8 \
        learning_rate=6e-4 \
        dataset='openwebtext' \
        max_iters=10 \
        wandb_log=False \
        gradient_accumulation_steps=4 \
        debug=True