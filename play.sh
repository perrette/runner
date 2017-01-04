#!/bin/bash
# do the work !

# global variables (settings)
ROOT=experiments_debug

# experiment specific settings
glacier=daugaard-jensen
exp=steadystate

# functions
# ---------

trim () { while read -r line; do echo "$line"; done; }

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
cmd="python play.py run $glacier $baseargs"
echo $cmd
eval $cmd

# run ensemble
# ============
cmd="python play.py run $glacier $baseargs"
echo $cmd
eval $cmd
