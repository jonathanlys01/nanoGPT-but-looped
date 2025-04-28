#!/bin/bash

#SBATCH --account=vaz@a100
#SBATCH --constraint=a100

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread

#SBATCH --job-name=fineweb-nanogpt
#SBATCH --output=slurm-logs/%j-%x.out
#SBATCH --error=slurm-logs/%j-%x.err
#SBATCH --time=03:00:00
#SBATCH --qos=qos_gpu_a100-t3
# Load the modules


echo "Job started at $(date)"
export WANDB_MODE=offline

module purge
module load arch/a100
module load pytorch-gpu/py3/2.6.0

mkdir -p slurm-logs
mkdir -p small_experiments

set -x
srun python train.py \
        out_dir="small_experiments/fineweb_baseline" \
        wandb_run_name="gpt2-baseline_logged" \
        model.n_encoder=0 \
        model.n_layer=12 \
        model.n_loop=1 \
        model.n_head=12 \
        model.n_embd=768 \
        init_from='scratch' \
        compile=True \
        batch_size=32 \
        learning_rate=6e-4 \
        dataset=fineweb-edu \
        max_iters=2_500 \
        lr_decay_iters=2_500 \
        gradient_accumulation_steps=128 \
        eval_interval=10 \
        eval_iters=50 \
        warmup_iters=100 \
        always_save_checkpoint=True