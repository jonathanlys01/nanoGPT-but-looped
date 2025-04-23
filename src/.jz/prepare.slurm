#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=prepost
#SBATCH --time=05:00:00
#SBATCH --account=vaz@v100
#SBATCH --cpus-per-task=8
#SBATCH --job-name=fw-prepro
#SBATCH --output=slurm-logs/fw-prepro.out
#SBATCH --error=slurm-logs/fw-prepro.err

# Load the modules      

echo "Job started at $(date)"

module purge
module load pytorch-gpu/py3/2.6.0

mkdir -p slurm-logs

srun python data/fineweb-edu/prepare.py \
        --num_proc 8 \
        --path $DSDIR/HuggingFace/HuggingFaceFW/fineweb-edu/sample/10BT \
        --cache_dir $SCRATCH/junk \
        --encoding gpt2
