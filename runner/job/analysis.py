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

from runner.register import register_job
from runner.prior import PriorParam
from runner.xrun import XRun, XData
from runner.job.config import load_config
from runner.job.model import model_parser, modelconfig, custommodel, \
    CustomModel, getmodel
from runner.job.run import parse_slurm_array_indices, _typechecker, run
from runner.job.run import XPARAM, EXPDIR, EXPCONFIG

XSTATE = "state.txt"
XLOGLIK = "loglik.txt" # log(weight)
LOGLIKS = "logliks.txt" # log-likelihood for various variables


analyze= argparse.ArgumentParser(parents=[modelconfig])
analyze.add_argument('expdir', default=EXPDIR, 
                               help='experiment directory to analyze')
analyze.add_argument('--out', default=None,
                               help='experiment directory to write the diagnostics to (by default same as expdir)')
analyze.add_argument('-i', '--in-state', 
                               help='input state file to consider (normally derived via custom getvar)')
grp =analyze.add_argument_group("model state", description='For now this requires a custom `getvar` function to retrieve state variables')
grp.add_argument("-v", "--state-variables", nargs='+', default=[],
                 help='list of state variables to include in state.txt, \
                 does not necessarily enter in the likelihood')
grp.add_argument('--stats', action='store_true', help='add statistics on model state')

grp =analyze.add_argument_group(
    "likelihood", 
    description='likelihood is provided a list of distributions (same convention as job sample) or via a custom `getcost`')

grp.add_argument('-l', '--likelihood',
                 type=PriorParam.parse,
                 help=PriorParam.parse.__doc__,
                 metavar="NAME=DIST",
                 default = [],
                 nargs='+')

grp.add_argument('--custom-cost', action='store_true',
                               help='use custom getcost function (adds up) \
                               (see runner.register.define)')


def getxrunanalysis(o, expdir):
    """get XRun instance for post-run analysis
    """
    paramsfile = os.path.join(expdir, XPARAM)
    cfg = load_config(os.path.join(expdir, EXPCONFIG))
    cfg["user_module"] = o.user_module
    model = getmodel(argparse.Namespace(**cfg), post_only=True) 
    assert model.executable is not None
    xparams = XData.read(paramsfile) # for the size & autodir
    return XRun(model, xparams, autodir=cfg["auto_dir"])


def analyze_post(o):

    xrun = getxrunanalysis(o, o.expdir)

    if not o.out:
        o.out = o.expdir

    # write state.txt
    # ===============
    names = o.state_variables + [x.name for x in o.likelihood]

    if o.in_state:
        print("Read state variables from",o.in_state)
        xstate = XData.read(o.in_state)
    elif names:
        print("Retrieve state variables from",o.expdir)
        xstate = xrun.getstate(names, o.expdir)
    else:
        xstate = None

    if xstate is not None:
        statefile = os.path.join(o.out, XSTATE)
        print("Write state variables to",statefile)
        xstate.write(statefile)

    # Derive likelihoods
    # ==================

    # direct distributions
    constraints = [l for l in o.likelihood]

    # Apply constraints on state 
    # ==========================
    logliks = np.empty((xrun.params.size, len(constraints)))

    for j, l in enumerate(constraints):
        jj = xstate.names.index(l.name)
        logliks[:, j] = l.dist.logpdf(xstate.values[:,jj])

    logliks[np.isnan(logliks)] = -np.inf
    xlogliks = XData(logliks, [l.name for l in constraints])

    file = os.path.join(o.out, LOGLIKS)
    print('write logliks to', file)
    xlogliks.write(file)

    # Sum-up and apply custom distribution
    # ====================================
    logliksum = logliks.sum(axis=1)

    if o.custom_cost:
        cost = xrun.getcost(o.out)
        logliksum += -0.5*cost

    file = os.path.join(o.out, "loglik.txt")
    print('write loglik (total) to', file)
    np.savetxt(file, logliksum)

    # Add statistics
    # ==============
    if not o.stats:
        return

    valid = np.isfinite(logliksum)
    ii = [xstate.names.index(c.name) for c in constraints]
    state = xstate.values[:, ii] # sort !
    pct = lambda p: np.percentile(state[valid], p, axis=0)

    names = [c.name for c in constraints]

    res = [
        ("obs", [c.dist.mean() for c in constraints]),
        ("best", state[np.argmax(logliksum)]),
        ("mean", state[valid].mean(axis=0)),
        ("std", state[valid].std(axis=0)),
        ("min", state[valid].min(axis=0)),
        ("p05", pct(5)),
        ("med", pct(50)),
        ("p95", pct(95)),
        ("max", state[valid].max(axis=0)),
    ]

    index = [nm for nm,arr in res if arr is not None]
    values = [arr for nm,arr in res if arr is not None]

    import pandas as pd
    df = pd.DataFrame(np.array(values), columns=names, index=index)

    with open(os.path.join(o.out, 'stats.txt'), 'w') as f:
        f.write(str(df))

register_job('analyze',analyze, analyze_post, 
             help="analyze ensemble (state + loglik + stats) for resampling")

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
