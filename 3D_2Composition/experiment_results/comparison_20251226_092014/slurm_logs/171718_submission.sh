#!/bin/bash

# Parameters
#SBATCH --job-name=submitit
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/home-mscluster/sfrasco/Masters-Research/3D_2Composition/experiment_results/comparison_20251226_092014/slurm_logs/%j_0_log.out
#SBATCH --partition=bigbatch
#SBATCH --signal=USR2@90
#SBATCH --time=4320
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /home-mscluster/sfrasco/Masters-Research/3D_2Composition/experiment_results/comparison_20251226_092014/slurm_logs/%j_%t_log.out /home-mscluster/sfrasco/miniconda3/envs/compneuro/bin/python -u -m submitit.core._submit /home-mscluster/sfrasco/Masters-Research/3D_2Composition/experiment_results/comparison_20251226_092014/slurm_logs
