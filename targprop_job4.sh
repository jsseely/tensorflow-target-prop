#!/bin/bash
#SBATCH --account=zi
#SBATCH --job-name=tprop4
#SBATCH -c 1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=1gb
#SBATCH --array=1-500
#SBATCH --error=./slurmerr/%A_%a.err
#SBATCH --output=./slurmout/%A_%a.out

python wrapper.py $SLURM_ARRAY_TASK_ID "170218_ae_tanh" "autoencoder" "tanh" 
