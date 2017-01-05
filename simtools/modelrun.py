#!/usr/bin/env python
"""Run model based on parameters defined in config.json

Note that they do not need to be the same as the original model, 
for instance parameters may be transform (e.g. inverse) to obtain
a more linear behaviour w.r.t the output.
"""
import os
import json
from collections import OrderedDict as odict
import numpy as np
import subprocess

def read_ensemble_params(pfile):
    pnames = open(pfile).readline().split()
    pvalues = np.loadtxt(pfile, skiprows=1)  
    return pnames, pvalues


def run_background(executable, cmd_args=(), ini_dir='.', logfile='out.out'):
    " execute in the background "
    print("Running job in background: %s" % (executable))
    print("...initial directory : %s" % (ini_dir))
    print("...log file : %s" % (logfile))
    cmd = " ".join(cmd_args) if not isinstance(cmd_args, basestring) else cmd_args
    #print "Storing output in: %s" % (out_dir)
    cmd = "'%s' %s > '%s' 2>&1 &" % (executable, cmd, logfile)
    if ini_dir != os.path.curdir:
        cmd = "cd '%s' && " % (ini_dir) + cmd  # go to initial directory prior to execution

    #print(cmd)
    code = os.system (cmd)

    return code

def run_foreground(executable, cmd_args=(), ini_dir='.', logfile=None):
    " execute in terminal, with blocking behaviour "
    cmd = " ".join(cmd_args) if not isinstance(cmd_args, basestring) else cmd_args
    cmd = "%s %s" % (executable, cmd)

    # execute from directory...
    if ini_dir != os.path.curdir:
        cmd = "cd '%s' && " % (ini_dir) + cmd

    if logfile is not None:
        cmd = cmd + " 2>&1 | tee "+logfile

    code = os.system (cmd)
    return code
