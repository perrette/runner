#!/usr/bin/env python
"""Main script to perform everything.
"""
from __future__ import print_function, absolute_import
import subprocess
import json
import sys
import os
import shutil
##sys.path.insert(0, 'scripts')
#from genparams import main as generate_params

GENPARAMS = "scripts/genparams.py"

def json_to_cmd(prior, out=None, scriptname=GENPARAMS):
    """Generate command-line argument from json dict
    """
    cmd = ["python", scriptname, "-p"]
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
    parser.add_argument("--force", "-f", action="store_true", 
                        help="erase experiment even if already existing")
    parser.add_argument("--root", default="experiments", 
                        help="experiments root directory (default=%(default)s)")
    parser.add_argument("--name", default="prior", 
                        help="experiment name (default=%(default)s)")
    parser.add_argument("--config", default="config.json", 
                        help="experiment config (default=%(default)s)")

    args = parser.parse_args(argv)

    expdir = os.path.join(args.root, args.name)
    config = json.load(open(args.config))

    # check if experiment already exists
    if os.path.exists(expdir) and os.listdir(expdir):
        if args.force:
            print("overwrite existing experiment")
        else:
            print("experiment already exist, exit")
            print("(use '-f' to proceed anyway)")
            sys.exit(1)

    if not os.path.exists(expdir):
        os.makedirs(expdir)

    # generate a Monte Carlo ensemble of parameters
    cmd = json_to_cmd(config["prior"], 
                      out=os.path.join(expdir, "params.txt"))

    print(" ".join(cmd))
    subprocess.call(cmd)

    # write command (for the record)
    with open(os.path.join(expdir, "command.sh"), "w") as f:
        f.write(" ".join(cmd))

if __name__ == "__main__":
    main()
