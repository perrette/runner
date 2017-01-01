#!/bin/bash
# do the work !

# generate parameters ensemble 
root=experiments
exp=steadystate
expdir=$root/$exp
params=$expdir/params.txt
GENPARAMS=scripts/genparams.py

if [ -f $params ] ; then
    echo "$params already exists. Delete? (y/n)"
    read ans
    if [ $ans != "y" ]; then
        exit 0
    fi
fi

cmd="python $GENPARAMS $(python parseconfig.py genparams --experiment $exp) --out $params"
mkdir -p $expdir
echo $cmd
eval $cmd
