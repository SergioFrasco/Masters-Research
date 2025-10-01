#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=4
#SBATCH --error=/home-mscluster/sfrasco/Masters-Research/2D_Partial_Observability/submitit_logs/%j_0_log.err
#SBATCH --job-name=hyperparam_sweep
#SBATCH --mem=16GB
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=/home-mscluster/sfrasco/Masters-Research/2D_Partial_Observability/submitit_logs/%j_0_log.out
#SBATCH --partition=bigbatch
#SBATCH --signal=USR2@90
#SBATCH --time=4320
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /home-mscluster/sfrasco/Masters-Research/2D_Partial_Observability/submitit_logs/%j_%t_log.out --error /home-mscluster/sfrasco/Masters-Research/2D_Partial_Observability/submitit_logs/%j_%t_log.err /home-mscluster/sfrasco/miniconda3/envs/compneuro/bin/python -u -m submitit.core._submit /home-mscluster/sfrasco/Masters-Research/2D_Partial_Observability/submitit_logs
