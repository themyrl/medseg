#!/bin/bash


JOBID=$(sbatch --parsable  test_depend.sh 0)

for i in $(seq 1 10);
do
  	JOBID=$(sbatch --parsable --dependency=afterany:${JOBID} test_depend.sh ${i})
done