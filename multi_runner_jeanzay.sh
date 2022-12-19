#!/bin/bash


JOBID=$(sbatch --parsable  one_runner_jeanzay.sh cv2)

for i in $(seq 3 5);
do
  	JOBID=$(sbatch --parsable --dependency=afterany:${JOBID} one_runner_jeanzay.sh cv${i})
  	# sbatch --parsable  one_runner_jeanzay.sh cv${i}
done