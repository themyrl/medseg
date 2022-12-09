#!/bin/bash
#SBATCH --job-name=fivelog/dataset     # job name
#SBATCH --ntasks=1                  # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=05:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=fivelog/dataset.out # output file name # add %j to id the job
#SBATCH --error=fivelog/dataset.err  # error file name # add %j to id the job
#   SBATCH -C v100-32g

set -x


cd $WORK/medseg
module purge
module load cuda/10.1.2
module load python/3.8.8




# Dataset preprocessing #dataset
srun python convert_dataset.py /gpfsscratch/rech/arf/unm89rb/WORD-V0.1.0/imagesTr /gpfsscratch/rech/arf/unm89rb/WORD-V0.1.0/imagesTr_npz
srun python convert_dataset.py /gpfsscratch/rech/arf/unm89rb/WORD-V0.1.0/labelsTr /gpfsscratch/rech/arf/unm89rb/WORD-V0.1.0/labelsTr_npz
srun python convert_dataset.py /gpfsscratch/rech/arf/unm89rb/WORD-V0.1.0/imagesVal /gpfsscratch/rech/arf/unm89rb/WORD-V0.1.0/imagesVal_npz
srun python convert_dataset.py /gpfsscratch/rech/arf/unm89rb/WORD-V0.1.0/labelsVal /gpfsscratch/rech/arf/unm89rb/WORD-V0.1.0/labelsVal_npz
