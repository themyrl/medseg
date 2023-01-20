#!/bin/bash
#SBATCH --job-name=worddbg     # job name
#SBATCH --ntasks=1                  # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=04:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=fivelog/worddbg.out # output file name # add %j to id the job
#SBATCH --error=fivelog/worddbg.err  # error file name # add %j to id the job
# # #     SBATCH -C v100-32g

set -x


cd $WORK/medseg
module purge
module load cuda/10.1.2
module load python/3.8.8




# Dataset preprocessing #dataset
# srun python convert_dataset_v2.py -i /gpfsscratch/rech/arf/unm89rb/LiverUS/imagesTr -o /gpfsscratch/rech/arf/unm89rb/LiverUS/imagesTr_npz -t float16
# srun python convert_dataset_v2.py -i /gpfsscratch/rech/arf/unm89rb/LiverUS/labelsStaple -o /gpfsscratch/rech/arf/unm89rb/LiverUS/labelsStaple_npz -t int16




# Training
# python mainWord.py -m model=nnunet dataset=word training=training_word #nnword





# Eval
python mainWord.py -m model=nnunet dataset=word training=training_word training.only_val=True #eval





