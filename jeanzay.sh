#!/bin/bash
#SBATCH --job-name=unetrdbg     # job name
#SBATCH --ntasks=1                  # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=00:10:00             # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_gpu-dev
#SBATCH --output=logs/unetrdbg.out # output file name # add %j to id the job
#SBATCH --error=logs/unetrdbg.err  # error file name # add %j to id the job
#   SBATCH -C v100-32g

set -x


cd $WORK/medseg
module purge
module load cuda/10.1.2
module load python/3.8.8




# Dataset preprocessing
# srun python convert_dataset.py /gpfsscratch/rech/arf/unm89rb/Trusted_v1_Loic/US_DATA/USimg /gpfsscratch/rech/arf/unm89rb/Trusted_v1_Loic/US_DATA/USimg_128_a 128 a
# srun python convert_dataset.py /gpfsscratch/rech/arf/unm89rb/Trusted_v1_Loic/US_DATA/USmask /gpfsscratch/rech/arf/unm89rb/Trusted_v1_Loic/US_DATA/USmask_128 128
# srun python convert_dataset.py /gpfsscratch/rech/arf/unm89rb/Trusted_v1_Loic/CT_DATA/CTimg /gpfsscratch/rech/arf/unm89rb/Trusted_v1_Loic/CT_DATA/CTimg_128_t 128 t
# srun python convert_dataset.py /gpfsscratch/rech/arf/unm89rb/Trusted_v1_Loic/CT_DATA/CTmask /gpfsscratch/rech/arf/unm89rb/Trusted_v1_Loic/CT_DATA/CTmask_128 128

# srun python convert_dataset.py /gpfsscratch/rech/arf/unm89rb/Trusted_v1_Loic/US_DATA/USmask_mf /gpfsscratch/rech/arf/unm89rb/Trusted_v1_Loic/US_DATA/USmask_mf_128 128 a
# srun python convert_dataset.py /gpfsscratch/rech/arf/unm89rb/Trusted_v1_Loic/US_DATA/USmask_a2 /gpfsscratch/rech/arf/unm89rb/Trusted_v1_Loic/US_DATA/USmask_a2_128 128 a
# srun python convert_dataset.py /gpfsscratch/rech/arf/unm89rb/Trusted_v1_Loic/US_DATA/USmask_a3 /gpfsscratch/rech/arf/unm89rb/Trusted_v1_Loic/US_DATA/USmask_a3_128 128 a
# srun python convert_dataset.py /gpfsscratch/rech/arf/unm89rb/Trusted_v1_Loic/CT_DATA/CTmask_mf /gpfsscratch/rech/arf/unm89rb/Trusted_v1_Loic/CT_DATA/CTmask_mf_128 128 t
# srun python convert_dataset.py /gpfsscratch/rech/arf/unm89rb/Trusted_v1_Loic/CT_DATA/CTmask_a2 /gpfsscratch/rech/arf/unm89rb/Trusted_v1_Loic/CT_DATA/CTmask_a2_128 128 t
# srun python convert_dataset.py /gpfsscratch/rech/arf/unm89rb/Trusted_v1_Loic/CT_DATA/CTmask_a3 /gpfsscratch/rech/arf/unm89rb/Trusted_v1_Loic/CT_DATA/CTmask_a3_128 128 t


# Training
# python mainDouble.py -m model=nnunet dataset=us_128_double_jz training=training_128_jz dataset.cv=cv1
# python mainDouble.py -m model=nnunet dataset=ct_128_double_jz training=training_128_jz dataset.cv=cv1
python mainDouble.py -m model=unetr dataset=ct_128_double_jz training=training_128_jz training.loss=Dice dataset.cv=cv1


# Evaluation
# srun python evaluator.py none /scratch/lthemyr/20220318_US_DATA/USmask_cropped /scratch/lthemyr/20220318_US_DATA/US_128/CROP_SMALL_nnu/NNUNET/ #evaluation : 4051
