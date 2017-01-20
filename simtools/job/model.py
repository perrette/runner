#!/usr/bin/env python
"""Prior parameter sampling
"""
from __future__ import print_function, division
import json
import inspect
import argparse
import sys
import numpy as np

from simtools.parsetools import CustomParser, Job
import simtools.sampling.resampling as xp
from simtools.xparams import XParams
from simtools.prior import Prior, GenericParam
import simtools.model.core as model
from simtools.model import filetypes, Model



# Model (for the initial setup)
# -----
model_parser = CustomParser(add_help=False)
grp = model_parser.add_argument_group("model config")

grp.add_argument("-x","--executable", help="model executable")

grp.add_argument("--args", default=[], nargs="*",
                 help="model arguments to replace whatever is in the config. Use the `--` separator to append arguments.)")

grp = model_parser.add_argument_group("params i/o")

grp.add_argument("--filetype", choices=filetypes.keys(), 
                 help="model params file type")
grp.add_argument("--default", 
                 help="default parameters for look-up and checking on parameter names. A file name with same format as specified by --filetype")

grp2 = grp.add_mutually_exclusive_group()
grp2.add_argument("--params-write", default="params.json",
                  help="param to write to each model rundir")
grp2.add_argument("--no-write", action="store_true",
                  help="do not write params to file")

grp2 = grp.add_mutually_exclusive_group()
grp2.add_argument("--params-args", default=None,
                  help="format for the command-line args (default=%(default)s)")
grp2.add_argument("--no-args", action="store_true",
                  help="do not pass param args via command-line")


# Programs
# ========
def show_config(prior, full=False, file=None, indent=None):
    """edit config w.r.t Prior Params and print to stdout
    """
    cfg = {
        "params": [json.loads(p.tojson()) for p in prior.params]
    }

    if full:
        if not file:
            print("ERROR: `--file FILE` must be provided with the `--full` option")
            sys.exit(1)
        full = json.load(open(file))
        full["prior"] = cfg
        cfg = full

    print(json.dumps(cfg, indent=indent, sort_keys=True))

show_config_main = ProgramParser(description=show_config.__doc_, parents=[prior_parser])

show_config.add_object(prior_parser, dest='prior')
show_config.add_argument("--full", action='store_true')
show_config.add_argument("--indent", type=int)
show_config.add_main(show_config)




def add_run(self):
    "run model"
    p = self.subparsers.add_parser('run', help=self.add_run.__doc__)
    p.add_argument("expdir", help="experiment directory (need to setup first)")

    p.add_argument("--id", type=int, help="run id")
    p.add_argument("--dry-run", action="store_true",
                   help="do not execute, simply print the command")
    p.add_argument("--background", action="store_true",
                   help="run in the background, do not check result")
    return p

def add_slurm_group(self, root):
    slurm = root.add_argument_group("slurm")
    slurm.add_argument("--qos", default="short", help="queue (default=%(default)s)")
    slurm.add_argument("--job-name", default=__file__, help="default=%(default)s")
    slurm.add_argument("--account", default="megarun", help="default=%(default)s")
    slurm.add_argument("--time", default="2", help="wall time m or d-h:m:s (default=%(default)s)")

def add_batch(self):
    "run ensemble"
    p = self.subparsers.add_parser("batch", 
                               help=self.add_batch.__doc__)
    p.add_argument("expdir", help="experiment directory (need to setup first)")

    #p.add_argument("--args", help="pass on to glacier")
    p.add_argument("--background", action="store_true", 
                      help="run in background instead of submitting to slurm queue")
    p.add_argument("--array",'-a', help="slurm sbatch --array")
    p.add_argument("--wait", action="store_true")
    self.add_slurm_group(p)
    return p





def main(argv=None):

    job = Job()
    job.add_command("config", show_config_main)
    job.add_command("product", product_main)
    job.add_command("sample", sample_main)
    job.add_command("resample", resample_main)
    job.add_command("neff", neff_main)
    #job.add_command("neff", Neff)

    job.main(argv)


if __name__ == "__main__":
    main()
