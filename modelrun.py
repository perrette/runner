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


GLACIER = "glacier"
SEC_IN_YEAR = 3600*24*365.25

def read_ensemble_params(pfile):
    pnames = open(pfile).readline().split()
    pvalues = np.loadtxt(pfile, skiprows=1)  
    return pnames, pvalues


def maybe_transform_param(name, value):
    """return param name and value understood by actual model
    """
    return name, value


def driving_stress(x, z, h, rho_i=910, g=9.81):
    return -rho_i*g*h[:-1]*(z[1:]-z[:-1])/(x[1:]-x[:-1])

def autoset_params(netcdf):
    import netCDF4 as nc
    ds = nc.Dataset(netcdf)

    tauc_max = driving_stress(ds["x"][:2], ds["surf"][:2], ds["H"][:2])[0]*1e-3
    assert tauc_max > 0, 'driving stress < 0, cant autoset sliding params, maybe smooth?'
    uq = ds["U"][0]*SEC_IN_YEAR
    h0 = ds["H"][0]

    ds.close()
    return tauc_max, uq, h0


def run_background(executable, cmd_args=(), ini_dir='.', logfile='out.out'):
    " execute in the background "
    print("Running job in background: %s" % (executable))
    print("...initial directory : %s" % (ini_dir))
    print("...log file : %s" % (logfile))
    cmd = " ".join(cmd_args) if not isinstance(cmd_args, basestring) else cmd_args
    #print "Storing output in: %s" % (out_dir)
    cmd = "'%s' %s > '%s' &" % (executable, cmd, logfile)
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

    if logfile is None:
        stdout = None
    else:
        # todo: add a log file
        stdout = None
        #stdout = file(logfile, 'w')

    code = subprocess.call(cmd, shell=True, stdout=stdout, stderr=subprocess.STDOUT)
    #print(cmd)
    #code = os.system (cmd)
    return code
