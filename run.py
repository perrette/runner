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

    parser.add_argument("--cmd", 
                        help="additional command-line arguments for glacier executable")

    parser.add_argument("--dry-run", action="store_true",
                        help="do not execute, simply print the command")

    #parser.add_argument("--autoset-sliding", action="store_true",
    #                    help="set tauc_max and uq based on driving stress and velocity at x = 0")
    #parser.add_argument("--ref-file", 
    #                    help="ref file to use e.g. to determine sliding params, by default same as input, but may be difference")

    args = parser.parse_args(argv)

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
    cmd = [args.exe, "--in_file", args.input, "--out_dir",args.out_dir]
    for k in params:
        name, value = maybe_transform_param(k, params[k])
        cmd.append("--"+name)
        cmd.append(str(value))

    cmdstr = " ".join(cmd)
    if args.cmd:
        cmdstr = cmdstr + " " + args.cmd

    print(cmdstr)
    
    if not args.dry_run:
        os.system(cmdstr)


if __name__ == "__main__":
    main()
