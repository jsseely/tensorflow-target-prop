#!/bin/bash
#SBATCH --account=zi
#SBATCH --job-name=tprop1
#SBATCH -c 1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2gb
#SBATCH --array=1-1000
#SBATCH --error=./slurmerr/%A_%a.err
#SBATCH --output=./slurmout/%A_%a.out

python wrapper.py $SLURM_ARRAY_TASK_ID "170209_cl_relu" "classification" "relu" 