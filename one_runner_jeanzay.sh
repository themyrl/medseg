#!/bin/bash
#SBATCH --job-name=sv2usglv2     # job name
#SBATCH --ntasks=1                  # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=48:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_gpu-t4
#SBATCH --output=multilogs/sv2usglv2%j.out # output file name # add %j to id the job
#SBATCH --error=multilogs/sv2usglv2%j.err  # error file name # add %j to id the job
# # #   SBATCH -C v100-32g

set -x


cd $WORK/medseg
module purge
module load cuda/10.1.2
module load python/3.8.8




# Dataset preprocessing
# srun python convert_dataset.py /gpfsscratch/rech/arf/unm89rb/Trusted_v1_Loic/US_DATA/USimg /gpfsscratch/rech/arf/unm89rb/Trusted_v1_Loic/US_DATA/USimg_128 128
# srun python convert_dataset.py /gpfsscratch/rech/arf/unm89rb/Trusted_v1_Loic/US_DATA/USmask /gpfsscratch/rech/arf/unm89rb/Trusted_v1_Loic/US_DATA/USmask_128 128


# Training
## one anno
# python main.py -m model=nnunet dataset=us_128_final_jz training=training_128_jz dataset.cv=$1
# python main.py -m model=cotr dataset=us_128_final_jz training=training_128_jz dataset.cv=$1

# python main.py -m model=nnunet dataset=ct_128_final_jz training=training_128_jz dataset.cv=$1
# python main.py -m model=cotr dataset=ct_128_final_jz training=training_128_jz dataset.cv=$1

## multi anno
# python mainDouble.py -m model=nnunet dataset=us_128_double_jz training=training_128_jz dataset.cv=$1 #m_usnn
# python mainDouble.py -m model=nnunet dataset=us_128_double_jz training=training_128_jz training.name=training_128_jz_gda dataset.cv=$1 #m_usnn_gda
# python mainDouble.py -m model=cotr dataset=us_128_double_jz training=training_128_jz dataset.cv=$1 #m_usco
# python mainDouble.py -m model=cotr dataset=us_128_double_jz training=training_128_jz training.name=training_128_jz_gda dataset.cv=$1 #m_usco_gda  !m_ctnn_gda
 
# python mainDouble.py -m model=nnunet dataset=ct_128_double_jz training=training_128_jz dataset.cv=$1 #m_ctnn
# python mainDouble.py -m model=nnunet dataset=ct_128_double_jz training=training_128_jz training.name=training_128_jz_gda dataset.cv=$1 #m_ctnn_gda

# python mainDouble.py -m model=cotr dataset=ct_128_double_jz training=training_128_jz dataset.cv=$1 #m_ctco
# python mainDouble.py -m model=cotr dataset=ct_128_double_jz training=training_128_jz training.name=training_128_jz_gda dataset.cv=$1 #m_ctco_gda


## multi anno v2
# python mainDoubleV2.py -m model=nnunet dataset=us_128_double_jz_v2 training=training_128_jz_v2 dataset.cv=$1 #v2usnn
# python mainDoubleV2.py -m model=nnunet dataset=us_128_double_jz_v2 training=training_128_jz_v2 dataset.cv=$1 training.checkpoint.load=True #cv2usnn
# python mainDoubleV2.py -m model=nnunet dataset=ct_128_double_jz_v2 training=training_128_jz_v2 dataset.cv=$1 #v2ctnn
# python mainDoubleV2.py -m model=nnunet dataset=ct_128_double_jz_v2 training=training_128_jz_v2 dataset.cv=$1 training.checkpoint.load=True #cv2ctnn

# python mainDoubleV2.py -m model=cotr dataset=us_128_double_jz_v2 training=training_128_jz_v2 dataset.cv=$1 #v2usco
# python mainDoubleV2.py -m model=cotr dataset=ct_128_double_jz_v2 training=training_128_jz_v2 dataset.cv=$1 #v2ctco


# python mainDoubleV2.py -m model=glam dataset=ct_128_double_jz_v2 training=training_128_jz_v2 dataset.cv=$1 #v2ctgl
# python mainDoubleV2.py -m model=glam dataset=us_128_double_jz_v2 training=training_128_jz_v2 dataset.cv=$1 #v2usgl


# python mainDoubleV2.py -m model=glam model.name=GLAMv2 dataset=ct_128_double_jz_v2 training=training_128_jz_v2 dataset.cv=$1 #v2ctglv2
# python mainDoubleV2.py -m model=glam model.name=GLAMv2 dataset=us_128_double_jz_v2 training=training_128_jz_v2 dataset.cv=$1 #v2usglv2


## simple anno v2
# python mainV2.py -m model=nnunet dataset=ct_128_simple_jz_v2 training=training_128_jz_v2 dataset.cv=$1 #sv2ctnn
# python mainV2.py -m model=nnunet dataset=us_128_simple_jz_v2 training=training_128_jz_v2 dataset.cv=$1 #sv2usnn

# python mainV2.py -m model=cotr dataset=ct_128_simple_jz_v2 training=training_128_jz_v2 dataset.cv=$1 #sv2ctco
# python mainV2.py -m model=cotr dataset=us_128_simple_jz_v2 training=training_128_jz_v2 dataset.cv=$1 #sv2usco

# python mainV2.py -m model=glam model.name=GLAMv2 dataset=ct_128_simple_jz_v2 training=training_128_jz_v2 dataset.cv=$1 #sv2ctglv2
python mainV2.py -m model=glam model.name=GLAMv2 dataset=us_128_simple_jz_v2 training=training_128_jz_v2 dataset.cv=$1 #sv2usglv2





### eval
# python mainDouble.py -m model=nnunet dataset=us_128_double_jz training=training_128_jz dataset.cv=$1 training.only_val=True #m_usnn
# python mainDouble.py -m model=cotr dataset=us_128_double_jz training=training_128_jz dataset.cv=$1 training.only_val=True #m_usco

# python mainDouble.py -m model=nnunet dataset=ct_128_double_jz training=training_128_jz dataset.cv=$1 training.only_val=True #m_ctnn
# python mainDouble.py -m model=cotr dataset=ct_128_double_jz training=training_128_jz dataset.cv=$1 training.only_val=True #m_ctco

# python mainV2.py -m model=cotr dataset=ct_128_simple_jz_v2 training=training_128_jz_v2 dataset.cv=$1 training.checkpoint.load=True training.only_val=True #evsv2ctco
# python mainV2.py -m model=nnunet dataset=us_128_simple_jz_v2 training=training_128_jz_v2 dataset.cv=$1 training.checkpoint.load=True training.only_val=True #evsv2usnn
# python mainV2.py -m model=nnunet dataset=ct_128_simple_jz_v2 training=training_128_jz_v2 dataset.cv=$1 training.checkpoint.load=True training.only_val=True #evsv2ctnn







