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

    tauc_max = driving_stress(ds["x"][:2], ds["surf"][:2], ds["H"][:2])[0]
    assert tauc_max > 0, 'driving stress < 0, cant autoset sliding params, maybe smooth?'
    uq = ds["U"][0]*SEC_IN_YEAR
    h0 = ds["H"][0]

    ds.close()
    return tauc_max, uq, h0


def run_background(executable, cmd_args=(), ini_dir='.', out_dir="."):
    " execute in the background "
    print("Running job in background: %s" % (executable))
    print("...initial directory : %s" % (ini_dir))
    print("...output directory : %s" % (out_dir))
    cmd = " ".join(cmd_args) if not isinstance(cmd_args, basestring) else cmd_args
    #print "Storing output in: %s" % (out_dir)
    cmd = "'%s' %s > '%s' &" % (executable, cmd, os.path.join(out_dir,"out.out"))
    if ini_dir != os.path.curdir:
        cmd = "cd '%s' && " % (ini_dir) + cmd  # go to initial directory prior to execution

    print(cmd)
    code = os.system (cmd)

    return code

def run_foreground(executable, cmd_args=(), ini_dir='.'):
    " execute in terminal, with blocking behaviour "
    cmd = " ".join(cmd_args) if not isinstance(cmd_args, basestring) else cmd_args
    cmd = "%s %s" % (executable, cmd)

    # execute from directory...
    if ini_dir != os.path.curdir:
        cmd = "cd '%s' && " % (ini_dir) + cmd

    print(cmd)
    code = os.system (cmd)
    return code


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", 
                        help="glacier netcdf input")
    parser.add_argument("--out-dir", required=True, help="output directory")
    parser.add_argument("--exe", default=GLACIER,
                        help="executable (default=%(default)s)")
    parser.add_argument("--config", default="config.json", 
                        help="default experiment config (default=%(default)s)")

    parser.add_argument("--file", '-i',
                        help="ensemble parameter file produced by params.txt")
    parser.add_argument("--id", type=int, 
                        help="experiment ID, required if --file is provided")
    parser.add_argument("--auto-subdir", action="store_true", 
                        help="subdirectory with run-id")

    parser.add_argument("--cmd", 
                        help="additional command-line arguments for glacier executable")

    parser.add_argument("--dry-run", action="store_true",
                        help="do not execute, simply print the command")

    parser.add_argument("--background", action="store_true",
                        help="run in the background, do not check result")


    args = parser.parse_args(argv)

    # create subdirectory to the output with the run-number
    if args.auto_subdir:
        assert args.id is not None
        runid = "{:0>5}".format(args.id)
        args.out_dir = os.path.join(args.out_dir, runid)

    # create full command line
    cfg = json.load(open(args.config))

    # first create a dictionary of parameters
    params = odict()
    
    # default arguments
    for k in sorted(cfg["default"].keys()):
        params[k] = cfg["default"][k]

    # data-dependent parameters
    tauc_max, uq, h0 = autoset_params(args.input)
    params["dynamics%tauc_max"] = tauc_max
    params["dynamics%Uq"] = uq
    params["dynamics%H0"] = h0

    # update arguments from file
    if args.file:
        pnames, pmatrix = read_ensemble_params(args.file)
        assert args.id is not None, "provide --id along with --file"
        pvalues = pmatrix[args.id]
        for k, v in zip(pnames, pvalues):
            params[k] = v

    # make command line argument for glacier executable
    cmd = ["--in_file", args.input, "--out_dir",args.out_dir]
    for k in params:
        name, value = maybe_transform_param(k, params[k])
        cmd.append("--"+name)
        cmd.append(str(value))

    cmdstr = " ".join(cmd)
    if args.cmd:
        cmdstr = cmdstr + " " + args.cmd

    print(args.exe, cmdstr)
    
    if not args.dry_run:
        #ret = os.system(cmdstr)
        if args.background:
            run_background(args.exe, cmd_args=cmdstr, out_dir=args.out_dir)

        else:
            ret = run_foreground(args.exe, cmd_args=cmdstr)
            # check return results
            if ret != 0:
                raise RuntimeError("Error when running the model")
            if not os.path.exists("simu_ok"):
                raise RuntimeError("simu_ok not written: error when running the model")

            print("Simulation successful")

if __name__ == "__main__":
    main()
