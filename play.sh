#!/bin/bash
# do the work !

# experiment specific settings
glacier=daugaard-jensen
exp=steadystate
size=100

# functions
# ---------
baseargs="--glacier $glacier --experiment $exp --size $size"

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
cmd="python play.py run $baseargs --background"
echo $cmd
eval $cmd

# run ensemble
# ============
cmd="python play.py runbatch $baseargs"
echo $cmd
eval $cmd
