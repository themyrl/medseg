#!/bin/bash
#SBATCH --job-name=debug     # job name
#SBATCH --ntasks=8                  # number of MP tasks
#SBATCH --ntasks-per-node=4          # number of MPI tasks per node
#SBATCH --gres=gpu:4                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=00:10:00             # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_gpu-dev
#----SBATCH --qos=qos_gpu-t4
#SBATCH --output=logs/debug.out # output file name # add %j to id the job
#SBATCH --error=logs/debug.err  # error file name # add %j to id the job
# # #   SBATCH -C v100-32g

set -x


cd $WORK/usseg
module purge
module load cuda/10.1.2
module load python/3.8.8


srun python main.py