#!/usr/bin/env python2.7
"""Play around with glacier model
"""
from __future__ import print_function, absolute_import
import argparse
import numpy as np
import json
import copy
import shutil
import os
import sys
import subprocess
from collections import OrderedDict as odict

#from glaciermodel import GlacierModel
#from simtools.modelrun import run_command, parse_slurm_array_indices
#from simtools.params import XParams, PriorParam, PriorParams
#from simtools.analysis import Likelihood, parse_constraint, Results
#from simtools.tools import parse_keyval
#from simtools.iis import IISExp
#from simtools.xrun import XRun
from simtools.register import register_job
from simtools.prior import PriorParam
from simtools.xrun import XParams, XRun, XDir
from simtools.job.config import _parser_defaults
from simtools.job.model import modelparser, modelconfig, CustomModel, getmodel
from simtools.job.run import parse_slurm_array_indices, _typechecker, run

# light version of job run 's simu group, to get to the output directory
simulight = argparse.ArgumentParser(add_help=False)
grp = simulight.add_argument_group("simulation settings")
grp.add_argument('--exp-dir', dest='expdir',
                  help='experiment directory')
grp.add_argument('-a','--auto-dir', action='store_true', 
                 help='must match `job run` settings')

grp.add_argument('-i','--params-file', 
                  help='ensemble parameters file (only for size), 
                 by default look for params.txt in the experiment directory')
#grp.add_argument('-j','--id', 
#                  type=_typechecker(parse_slurm_array_indices), dest='runid', 
#                 metavar="I,J...,START-STOP:STEP,...", 
#                  help='same as job run, by default take all')
#grp.add_argument('--include-default', 
#                  action='store_true', 
#                  help='also analyze default model version')

# also include all default values from run, for getmodel etc. to work
defs = _parser_defaults(run)
simulight.set_defaults(defs)

## add other parameters but hide them, so that we can use getmodel()
#defined = [a.dest for a in simulight._actions]
#for a in modelparser._actions:
#    if a.dest not in defined:
#        a2 = copy.copy(a)  # need copy otherwise affects modelparser
#        a2.help = argparse.SUPPRESS
#        simulight._add_action(a2)

state = argparse.ArgumentParser(add_help=False, parents=[])
grp = likelihood.add_argument_group("model state")
grp.add_argument("-v", "--state-variables", nargs='+', default=[],
                 help='list of state variables, at least including the constraints \
                 (default is to stick to constraint variables, if provided)')

likelihood = argparse.ArgumentParser(add_help=False, parents=[])
grp = likelihood.add_argument_group("likehood from obs constraints")
grp.add_argument('-l', '--likelihood',
                 type=PriorParam.parse,
                 help=PriorParam.parse.__doc__,
                 metavar="NAME=DIST",
                 default = [],
                 nargs='+')

writestate_parser = argparse.ArgumentParser(add_help=False, 
                                            parents=[simulight, state, likelihood],
                                            description='derive state variables')
writestate_parser.add_argument('--state-file', 
                               help='default to state.txt under teh experiment dir')

def writestate(o):

    # determine name of state variables to consider
    if not o.state_variables and not o.likelihood:
        writestate_parser.error("need to provide either state variables or likelihood")

    names = o.state_variables + [l.name for l in o.likelihood 
                                 if l.name not in o.state_variables]

    paramsfile = o.params_file or os.path.join(o.expdir, 'params.txt')
    statefile = o.state_file or os.path.join(o.expdir, 'state.txt')

    model = getmodel(o) 
    # qui peut le plus peut le moins
    xparams = Xparams.read(paramsfile) # for the size & autodir
    xrun = XRun(model, xparams, autodir=o.autodir)
    xstate = xrun.getstate(names, o.expdir)
    print("Write state variables to",statefile)
    xstate.write(statefile)

register_job('state', writestate_parser, writestate, help=writestate_parser.__doc__)

#grp.add_argument('-m', '--module', 
#                 help='module file where ')

#likelihood.add_argument('--')

#def likelihood_post(o):


#    def add_loglik(self):
#        p =  self.subparsers.add_parser("loglik", 
#                               help="return log-likelihood for one run")
#        p.add_argument("expdir", help="experiment directory (need to setup first)")
#        p.add_argument("--id", type=int, help='specify only on run')
#        p.add_argument("-l", "--constraints-file", 
#                       help="constraints to compute likelihood")
#        return p
#
#    def add_constraints_group(self, subp):
#        grp = subp.add_argument_group("obs constraints")
#        grp.add_argument("--obs-file", help="obs constraints config file")
#        grp.add_argument("--obs", nargs='*', default=[], help="list of obs constraints")
#
#    def add_analysis(self):
#        """analysis for the full ensemble: state, loglik, etc...
#        """
#        subp = self.subparsers.add_parser("analysis", help=self.add_analysis.__doc__)
#        subp.add_argument("expdir", help="experiment directory (need to setup first)")
#        self.add_constraints_group(subp)
#        subp.add_argument('-f', '--force', action='store_true',
#                       help='force analysis even if loglik.txt already present')
#        return subp
#
#    def add_iis(self):
#        """run a number of iterations following IIS methodology
#        """
#        # perform IIS optimization
#        subp = self.subparsers.add_parser("iis", parents=[parent], 
#                                   help=self.add_iis.__doc__)
#        subp.add_argument("expdir", help="experiment directory (need to setup first)")
#        self.add_constraints_group(subp)
#        subp.add_argument("-n", "--maxiter", type=int, required=True, 
#                          help="max number of iterations to reach")
#        subp.add_argument("--start", type=int, default=0,
#                          help="start from iteration (default=0), note: previous iter must have loglik.txt file")
#        subp.add_argument("--restart", action='store_true', 
#                          help="automatically find start iteration")
#        subp.add_argument("--epsilon", default=None, type=float, 
#                help="loglik weight + jitter")
#        return subp
#
#    def parse_args(self, *args, **kwargs):
#        return self.parser.parse_args(*args, **kwargs)
#
#
##def get_constraints(args, getobs):
##    like = Likelihood.read(args.obs_file, getobs)
##    constraints = [parse_constraint(cstring, getobs=getobs) 
##                   for cstring in args.obs]
##    like.update(constraints)
##    return like.constraints
#
#
##    elif args.cmd == "analysis":
##
##        # model config & params already present
##        print("analysis of experiment", args.expdir)
##        xrun = XRun.read(args.expdir)
##
##        if os.path.exists(xrun.path("loglik.txt")) and not args.force:
##            raise ValueError("analysis already performed, use --force to overwrite")
##
##        # define constraints
##        constraints = get_constraints(args, xrun.model.getobs)
##
##        # analyze
##        results = xrun.analyze(constraints)
##        results.write(args.expdir)
##
##
##    elif args.cmd == "iis":
##
##        constraints = get_constraints(args, xrun.model.getobs)
##
##        iis = IISExp(args.expdir, constraints, iter=args.start, epsilon=args.epsilon, 
##                     resampling=args.resampling_method)
##
##        if args.restart:
##            iis.goto_last_iter()
##        iis.runiis(args.maxiter)
#
#    else:
#        raise NotImplementedError("subcommand not yet implemented: "+args.cmd)
#
#
#if __name__ == '__main__':
#    main()
