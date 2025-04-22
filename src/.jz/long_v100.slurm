#!/bin/bash

#SBATCH --account=vaz@v100
#SBATCH --partition=gpu_p2

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=3
#SBATCH --hint=nomultithread

#SBATCH --job-name=looped-nanogpt
#SBATCH --output=slurm-logs/%j-%x.out
#SBATCH --error=slurm-logs/%j-%x.err
#SBATCH --time=00:30:00
#SBATCH --qos=qos_gpu-dev
# Load the modules


echo "Job started at $(date)"
export WANDB_MODE=offline

module purge
module load pytorch-gpu/py3/2.6.0

mkdir -p slurm-logs
mkdir -p small_experiments

set -x
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
        max_iters=50_000 \
        lr_decay_iters=50_000 \
        wandb_log=False \
        dtype=float16 \
        compile=True \
        model.use_self_attention=False
# flash attn is not supported on v100