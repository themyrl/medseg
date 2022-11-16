#!/bin/bash
#SBATCH --job-name=debug     # job name
#SBATCH --ntasks=1                  # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=01:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=logs/debug.out # output file name # add %j to id the job
#SBATCH --error=logs/debug.err  # error file name # add %j to id the job
# # #   SBATCH -C v100-32g

set -x


cd $WORK/usseg
module purge
module load cuda/10.1.2
module load python/3.8.8




# Dataset preprocessing
srun python convert_dataset.py /gpfsscratch/rech/arf/unm89rb/Trusted_v1_Loic/US_DATA/USimg /gpfsscratch/rech/arf/unm89rb/Trusted_v1_Loic/US_DATA/USimg_128 128
srun python convert_dataset.py /gpfsscratch/rech/arf/unm89rb/Trusted_v1_Loic/US_DATA/USmask /gpfsscratch/rech/arf/unm89rb/Trusted_v1_Loic/US_DATA/USmask_128 128


# Training
# srun python main.py