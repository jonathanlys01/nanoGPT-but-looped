#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=prepost
#SBATCH --time=05:00:00
#SBATCH --account=vaz@v100
#SBATCH --job-name=owt-prepro
#SBATCH --output=slurm-logs/owt-prepro.out
#SBATCH --error=slurm-logs/owt-prepro.err

# Load the modules      

echo "Job started at $(date)"

module purge
module load pytorch-gpu/py3/2.6.0

mkdir -p slurm-logs

srun python data/openwebtext/prepare.py \
        --num_proc 8 \
        --path $DSDIR/HuggingFace/Skylion007/openwebtext \
        --cache_dir $SCRATCH/junk \
        --encoding gpt2
