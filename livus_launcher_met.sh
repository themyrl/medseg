#!/bin/bash
#SBATCH -A arf@v100
#SBATCH --job-name=data     # job name
#SBATCH --ntasks=1                  # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=19:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=fivelog/data.out # output file name # add %j to id the job
#SBATCH --error=fivelog/data.err  # error file name # add %j to id the job
#SBATCH -C v100-32g

set -x


cd $WORK/medseg
module purge
module load cuda/10.1.2
module load python/3.8.8




# Dataset preprocessing #dataset
# srun python convert_dataset_v2.py -i /gpfsscratch/rech/arf/unm89rb/LiverUS/imagesTr -o /gpfsscratch/rech/arf/unm89rb/LiverUS/imagesTr_npz -t float16
# srun python convert_dataset_v2.py -i /gpfsscratch/rech/arf/unm89rb/LiverUS/labelStaple -o /gpfsscratch/rech/arf/unm89rb/LiverUS/labelStaple_npz -t int16
srun python convert_dataset_v2.py -i /gpfsscratch/rech/arf/unm89rb/Trusted_v1_Loic/US_DATA/USimg -o /gpfsscratch/rech/arf/unm89rb/Trusted_v1_Loic/US_DATA/USimg_128_t -s 128 -m trilinear -t float32

srun python convert_dataset_v2.py -i /gpfsscratch/rech/arf/unm89rb/Trusted_v1_Loic/US_DATA/USmask_mf -o /gpfsscratch/rech/arf/unm89rb/Trusted_v1_Loic/US_DATA/USmask_mf_128 -s 128 -m trilinear -t int16
srun python convert_dataset_v2.py -i /gpfsscratch/rech/arf/unm89rb/Trusted_v1_Loic/US_DATA/USmask_a2 -o /gpfsscratch/rech/arf/unm89rb/Trusted_v1_Loic/US_DATA/USmask_a2_128 -s 128 -m trilinear -t int16
srun python convert_dataset_v2.py -i /gpfsscratch/rech/arf/unm89rb/Trusted_v1_Loic/US_DATA/USmask_a3 -o /gpfsscratch/rech/arf/unm89rb/Trusted_v1_Loic/US_DATA/USmask_a3_128 -s 128 -m trilinear -t int16



# Training
# python mainLivus.py -m model=nnunet dataset=livus training=training_livus #livus


# Eval
# python mainLivus.py -m model=nnunet dataset=us_128_double_jz_v2 training=training_livus training.only_val=True #evallivus





