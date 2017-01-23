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





def modelrun(self):
    "run model"
    parser = CustomParser(parents=[model_parser], 
                          description=modelconfig.__doc__)
    parser.add_argument("--config-file", 
                        help='configuration file')
    p = .add_parser('run', help=self.add_run.__doc__)
    p.add_argument("expdir", help="experiment directory (need to setup first)")

    p.add_argument("--id", type=int, help="run id")
    p.add_argument("--dry-run", action="store_true",
                   help="do not execute, simply print the command")
    p.add_argument("--background", action="store_true",
                   help="run in the background, do not check result")
    return p
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
