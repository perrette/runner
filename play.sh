#!/bin/bash
# do the work !

# global variables (settings)
ROOT=experiments
GENPARAMS=scripts/genparams.py
config=config.json

# experiment specific settings
glacier=daugaard-jensen
exp=steadystate

# dependent variables
expdir=$ROOT/$exp
params=$expdir/prior.txt

# functions
# ---------

trim () { while read -r line; do echo "$line"; done; }


# generate parameter ensemble
# ---------------------------
if [ -f $params ] ; then
    #echo "$params already exists. Delete? (y/n)"
    echo "$params already exists. Re-use it."
else
    cmd="python $GENPARAMS $(python parseconfig.py genparams --experiment $exp --config $config) --out $params"
    mkdir -p $expdir
    echo $cmd
    echo $cmd > $expdir/command-genparams.sh
    eval $cmd
fi

# run ensemble
# ============

N=$(wc $params | trim | cut -d" " -f1)  # number of runs
priordir=$expdir/prior
args="glaciers/$glacier.nc --config $config --experiment $exp --file $params --out-dir $priordir --auto-subdir"
#cmd="SLURM_ARRAY_TASK_ID=0 ./job.sh $args"
cmd="sbatch --array=0-$((N-1)) ./job.sh $args"
mkdir -p logs
echo $cmd
eval $cmd
