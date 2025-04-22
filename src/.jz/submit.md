# SLURM dependency submission

The following command submits the small job first, gets its job ID and then submits the long job with a dependency on the small job. The long job will only start after the small job has completed successfully.

```bash
sbatch --dependency=afterok:$(sbatch ./jz/small_a100.slurm | awk '{print $4}') ./jz/long_a100.slurm
```