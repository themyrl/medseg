#!/bin/bash
#SBATCH --job-name=eval     # job name
#SBATCH --ntasks=1                  # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=12:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_gpu-t4
#SBATCH --output=logs/eval.out # output file name # add %j to id the job
#SBATCH --error=logs/eval.err  # error file name # add %j to id the job
#SBATCH -C v100-32g

set -x


cd $WORK/medseg
module purge
module load cuda/10.1.2
module load python/3.8.8




# Dataset preprocessing
# srun python convert_dataset.py /gpfsscratch/rech/arf/unm89rb/Trusted_v1_Loic/US_DATA/USimg /gpfsscratch/rech/arf/unm89rb/Trusted_v1_Loic/US_DATA/USimg_128 128
# srun python convert_dataset.py /gpfsscratch/rech/arf/unm89rb/Trusted_v1_Loic/US_DATA/USmask /gpfsscratch/rech/arf/unm89rb/Trusted_v1_Loic/US_DATA/USmask_128 128
# srun python convert_dataset.py /gpfsscratch/rech/arf/unm89rb/Trusted_v1_Loic/CT_DATA/CTimg /gpfsscratch/rech/arf/unm89rb/Trusted_v1_Loic/CT_DATA/CTimg_128 128
# srun python convert_dataset.py /gpfsscratch/rech/arf/unm89rb/Trusted_v1_Loic/CT_DATA/CTmask /gpfsscratch/rech/arf/unm89rb/Trusted_v1_Loic/CT_DATA/CTmask_128 128


# Training
# python main.py -m model=nnunet dataset=us_128_final_jz training=training_128_jz dataset.cv=cv1


# Evaluation
srun python evaluator.py none /scratch/lthemyr/20220318_US_DATA/USmask_cropped /scratch/lthemyr/20220318_US_DATA/US_128/CROP_SMALL_nnu/NNUNET/ #evaluation : 4051
