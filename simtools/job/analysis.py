#!/usr/bin/env python2.7
"""Play around with glacier model
"""
from __future__ import print_function, absolute_import, division
import argparse
import numpy as np
import json
import copy
import shutil
import os
import sys
import subprocess
from collections import OrderedDict as odict
from scipy.stats import norm

#from glaciermodel import GlacierModel
#from simtools.modelrun import run_command, parse_slurm_array_indices
#from simtools.params import XParams, PriorParam, PriorParams
#from simtools.analysis import Likelihood, parse_constraint, Results
#from simtools.tools import parse_keyval
#from simtools.iis import IISExp
#from simtools.xrun import XRun
from simtools.register import register_job
from simtools.prior import PriorParam
from simtools.xrun import XParams, XRun, XState, DataFrame
from simtools.job.config import load_config
from simtools.job.model import model_parser, modelconfig, custommodel, \
    CustomModel, getmodel
from simtools.job.run import parse_slurm_array_indices, _typechecker, run
from simtools.job.run import XPARAM, EXPDIR, EXPCONFIG

XSTATE = "state.txt"
XLOGLIK = "loglik.txt" # log-likelihood for various variables
XWEIGHT = "weights.txt"

state = argparse.ArgumentParser(add_help=False, parents=[])
grp = state.add_argument_group("model state")
grp.add_argument("-v", "--state-variables", nargs='+', default=[],
                 help='list of state variables, at least including the constraints \
                 (default is to stick to constraint variables, if provided)')

writestate_parser = argparse.ArgumentParser(add_help=False, 
                                            parents=[state, custommodel],
                                            description='Derive state variables.\
                                            of a previous experiment.')
writestate_parser.add_argument('expdir', default=EXPDIR, 
                               help='experiment directory (state will be written there)')
writestate_parser.add_argument('--state-file', 
                               help='default to '+XSTATE+' under the experiment dir')


def getxrunanalysis(o):
    """get XRun instance for post-run analysis
    """
    paramsfile = os.path.join(o.expdir, XPARAM)
    cfg = load_config(os.path.join(o.expdir, EXPCONFIG))
    cfg.update(vars(o))
    model = getmodel(argparse.Namespace(**cfg), post_only=True) 
    xparams = XParams.read(paramsfile) # for the size & autodir
    return XRun(model, xparams, autodir=cfg["auto_dir"])


def writestate(o):

    # determine name of state variables to consider
    assert o.state_variables, \
        writestate_parser.error("requires state variables -v/--state-variables")

    names = o.state_variables 

    xrun = getxrunanalysis(o)

    xstate = xrun.getstate(names, o.expdir)
    statefile = o.state_file or os.path.join(o.expdir, XSTATE)
    print("Write state variables to",statefile)
    xstate.write(statefile)


register_job('state', writestate_parser, writestate, help="derive state variables")


likelihood_parser = argparse.ArgumentParser(add_help=False, 
                                            parents=[writestate_parser],
                                            description='Compute likelihood.')

grp = likelihood_parser.add_argument_group("likehood from obs constraints")
grp.add_argument('-l', '--likelihood',
                 type=PriorParam.parse,
                 help=PriorParam.parse.__doc__,
                 metavar="NAME=DIST",
                 default = [],
                 nargs='+')


class Obs(object):
    " error as normal dist around obs "
    def __init__(self, name, err, pct=False):
        self.name = name
        self.err = err
        self.pct = pct

    @classmethod
    def parse(cls, string):
        "observation error in absolute (NAME=STD) or relative (NAME=STD%) term"
        name, spec = string.split('=')
        if spec.endswith('%'):
            err = float(spec[:-1])
            pct = True
        else:
            err = float(spec)
            pct = False
        return cls(name, err, pct)

    def __str__(self):
        if self.pct:
            pattern = "{}={}%"
        else:
            pattern = "{}={}"
        return pattern.format(self.name, self.err)

    def get_dist(self, mean):
        " return Likelihood type given mean"
        if self.pct:
            dist = norm(mean, self.err*self.pct/100.)
        else:
            dist = norm(mean, self.err)
        return PriorParam(self.name, dist)


grp.add_argument('--obs-error',
                 type=Obs.parse,
                 help=Obs.parse.__doc__,
                 metavar="NAME=ERR",
                 default = [],
                 nargs='+')


likelihood_parser.add_argument('--weights-file', 
                               help='final likelihood default to '+XWEIGHT+' under the experiment dir')
likelihood_parser.add_argument('--loglik-file', 
                               help='log-like matrix of individual constraints, default to '+XLOGLIK+' under exp dir')

#analyze_parser.add_argument()

def getstate(o):
    statefile = o.state_file or os.path.join(o.expdir, XSTATE)
    return XState.read(statefile)


def likelihood_post(o):
    state = getstate(o)

    # direct distributions
    likelihood = [l for l in o.likelihood]

    # build likelihood from obs-errors (requires getobs)
    if o.obs_error:
        # qui peut le plus peut le moins
        model = getxrunanalysis(o).model
        # TODO: add fromjson to Model class, to avoid messing global variables
        for err in o.obs_error:
            mean = model.getobs(err.name)
            like = Obs.get_dist(mean)
            likelihood.append(like)

    assert likelihood, 'requires -l/--likelihood OR --obs-error'
    loglik = np.empty((state.size, len(likelihood)))

    for j, l in enumerate(likelihood):
        jj = state.names.index(l.name)
        loglik[:, j] = l.logpdf(state.values[:,jj])

    loglik[np.isnan(loglik)] = -np.inf

    xloglik = DataFrame(loglik, [l.name for l in likelihood])
    file = o.weights_file or os.path.join(o.expdir, XLOGLIK)
    print('write loglik to', file)

    weights = np.exp(loglik.sum(axis=1))
    file = o.loglik_file or os.path.join(o.expdir, XLOGLIK)
    print('write weights to', file)
    np.savetxt(weights, file)


register_job('likelihood', likelihood_parser, likelihood_post, 
             help="derive likelihood weights from constraints")

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
