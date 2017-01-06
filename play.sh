#!/bin/bash
# do the work !

# global variables (settings)
ROOT=experiments_6

# experiment specific settings
glacier=daugaard-jensen
exp=steadystate

# functions
# ---------
baseargs="--experiment $exp --expdir $ROOT/$exp"

# generate parameter ensemble
# ===========================
cmd="python play.py genparams $baseargs"
echo $cmd
eval $cmd

if [ $? != 0 ]; then
    exit 1
fi

# run default
# ===========
cmd="python play.py run $glacier $baseargs --background"
echo $cmd
eval $cmd

# run ensemble
# ============
cmd="python play.py runbatch $glacier $baseargs"
echo $cmd
eval $cmd
