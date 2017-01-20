#!/usr/bin/env python
"""Model simulations
"""
from __future__ import print_function, division
import json
import inspect
import argparse
import sys
import numpy as np

from simtools.parsetools import CustomParser
#import simtools.sampling.resampling as xp
from simtools.xparams import XParams
#from simtools.prior import Prior, GenericParam
import simtools.model.core as model
from simtools.model import filetypes, Model, Param
from simtools.model.params import print_filetypes

from .prior import prior_parser



# Model (for the initial setup)
# -----
model_parser = CustomParser(add_help=False)
grp = model_parser.add_argument_group("model config")

grp.add_argument("-x","--executable", help="model executable")

grp.add_argument("--args", default=[], nargs="*",
                 help="model arguments to replace whatever is in the config. Use the `--` separator to append arguments.)")

grp.add_argument("--params-file", 
                 help="(optional) default parameters for look-up and checking on parameter names. A file name with same format as specified by --filetype")
grp.add_argument("-p", "--params-default", nargs="*", 
                 type=Param.parse, metavar="NAME=VALUE", default=[], 
                 help="command-line version of --params-file")

grp = model_parser.add_argument_group("params i/o")

grp.add_argument("--filetype",                  
                 help="model params file type")
grp.add_argument("--template",                  
                 help="along with `--filetype template`, this file is a template parameter file with '{NAME}' fields to be formatted with actual parameter values. This option also requires `--params-default` to be provided via command-line since a template format is write-only (to pass perturbed parameters to model, if required)")

grp.add_argument("--params-write", default="params.json",
                  help="param to write to each model rundir (do not write if left empty")
#grp2.add_argument("--no-write", action="store_true",
#                  help="do not write params to file")

grp.add_argument("--params-args", default=None,
                  help="format for the command-line args, e.g. '--{name} {value}'. No parameter will be passed to model if left empty.")
#grp2.add_argument("--no-args", action="store_true",
#                  help="do not pass param args via command-line")


# Programs
# ========
def modelconfig(argv=None):
    "setup model configuration via command-line"
    parser = CustomParser(parents=[model_parser], 
                          description=modelconfig.__doc__)
    parser.add_argument("--config-file", 
                        help='general configuration file to include in the output')
    parser.add_argument("--indent", type=int, help='json output')
    o = parser.parse_args(argv)
    model = Model(o.executable, o.args, o.params_default, o.params_args, o.params_write, o.filetype)

    jsonstring = model.tojson()

    if o.config_file:
        cfg = json.load(open(o.config_file))
        cfg.update(json.loads(jsonstring))
        jsonstring = json.dumps(cfg, sort_keys=True, indent=o.indent)

    print(jsonstring)


#def add_run(self):
#    "run model"
#    p = self.subparsers.add_parser('run', help=self.add_run.__doc__)
#    p.add_argument("expdir", help="experiment directory (need to setup first)")
#
#    p.add_argument("--id", type=int, help="run id")
#    p.add_argument("--dry-run", action="store_true",
#                   help="do not execute, simply print the command")
#    p.add_argument("--background", action="store_true",
#                   help="run in the background, do not check result")
#    return p
#
#def add_slurm_group(self, root):
#    slurm = root.add_argument_group("slurm")
#    slurm.add_argument("--qos", default="short", help="queue (default=%(default)s)")
#    slurm.add_argument("--job-name", default=__file__, help="default=%(default)s")
#    slurm.add_argument("--account", default="megarun", help="default=%(default)s")
#    slurm.add_argument("--time", default="2", help="wall time m or d-h:m:s (default=%(default)s)")
#
#def add_batch(self):
#    "run ensemble"
#    p = self.subparsers.add_parser("batch", 
#                               help=self.add_batch.__doc__)
#    p.add_argument("expdir", help="experiment directory (need to setup first)")
#
#    #p.add_argument("--args", help="pass on to glacier")
#    p.add_argument("--background", action="store_true", 
#                      help="run in background instead of submitting to slurm queue")
#    p.add_argument("--array",'-a', help="slurm sbatch --array")
#    p.add_argument("--wait", action="store_true")
#    self.add_slurm_group(p)
#    return p
