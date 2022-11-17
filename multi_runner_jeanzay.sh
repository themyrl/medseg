#!/bin/bash


JOBID=$(sbatch --parsable  one_runner_jeanzay.sh cv1)

for i in $(seq 2 5);
do
  	JOBID=$(sbatch --parsable --dependency=afterany:${JOBID} one_runner_jeanzay.sh cv${i})
done