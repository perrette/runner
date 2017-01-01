#!/usr/bin/env python
"""Parse config file to get appropriate command line arguments for various general-purpose scripts.
"""
from __future__ import print_function, absolute_import
import json

def get_expconfig(config, expname):
    expnames = [expconfig["name"] for expconfig in config["experiments"]]
    try:
        i = expnames.index(expname)
    except:
        print("available experiments: ", expnames)
        raise

    expconfig = config["experiments"][i]
    return expconfig

def genparams_args(prior, out=None):
    """Generate command-line argument from json dict
    """
    cmd = ["--params"]
    for p in prior["params"]:
        lo, up = p["range"]
        cmd.append("{}=uniform?{},{}".format(p["name"],lo, up))

    cmd.append("--mode={}".format(prior["sampling"]))
    cmd.append("--size={}".format(prior["size"]))
    if prior["seed"] > 0:
        cmd.append("--seed={}".format(prior["seed"]))

    if out is not None:
        cmd.append("--out="+out)
    return cmd

def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='cmd')

    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--config", default="config.json", 
                        help="experiment config (default=%(default)s)")
    parent.add_argument("--experiment", required=True, #default="steadystate", 
                        help="experiment name") # (default=%(default)s)")
    parent.add_argument("--glacier", default="daugaard-jensen", 
                        help="glacier name (default=%(default)s)")
    parent.add_argument("--root", default="experiments", 
                        help="experiments root directory (default=%(default)s)")

    p1 = subparsers.add_parser('genparams', parents=[parent], 
                               help="generate ensemble")

    args = parser.parse_args(argv)

    config = json.load(open(args.config))

    expconfig = get_expconfig(config, args.experiment)

    if args.cmd == "genparams":
        cmd = genparams_args(expconfig["prior"])
    else:
        raise NotImplementedError("subcommand not yet implemented: "+args.cmd)

    print(" ".join(cmd))

if __name__ == "__main__":
    main()
