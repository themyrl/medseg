#!/bin/bash


JOBID=$(sbatch --parsable  one_runner_jeanzay.sh cv1)
# JOBID=$(sbatch --parsable  one_runner_jeanzay.sh cv2)
# JOBID=$(sbatch --parsable  one_runner_jeanzay.sh cv3)
# JOBID=$(sbatch --parsable  one_runner_jeanzay.sh cv4)
# JOBID=$(sbatch --parsable  one_runner_jeanzay.sh cv5)

# for i in $(seq 2 5);
# do
#   	# JOBID=$(sbatch --parsable --dependency=afterany:${JOBID} one_runner_jeanzay.sh cv${i})
#   	sbatch --parsable  one_runner_jeanzay.sh cv${i}
# done