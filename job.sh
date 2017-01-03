#!/bin/bash

#SBATCH --qos=short
#SBATCH --job-name=glacier-slr-potential
#SBATCH --account=megarun
#SBATCH --output=logs/log-%A-%a.out
#SBATCH --error=logs/log-%A-%a.err

echo
echo SLURM JOB
echo ---------
echo "SLURM_JOBID $SLURM_JOBID"
echo "SLURM_ARRAY_JOB_ID $SLURM_ARRAY_JOB_ID"
echo "SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID"

#cmd="python run.py $@ --id $SLURM_ARRAY_TASK_ID"
cmd="python play.py run $@ --id $SLURM_ARRAY_TASK_ID"
echo $cmd
eval $cmd
