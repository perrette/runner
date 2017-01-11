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

from glaciermodel import GlacierModel
from simtools.modelrun import run_command, parse_slurm_array_indices
from simtools.genparams import PriorParam, PriorParams, XParams
from simtools.costfunction import Likelihood, parse_constraint
from simtools.tools import str_dataframe, parse_keyval
from simtools.resample import Resampler
import netCDF4 as nc

# default directory structure
glaciersdir = "glaciers"
constraintsdir = "config.like"
priorsdir = "config.prior"
configsdir = "config.model"
experimentsdir = "experiments"

class ExpDir(object):
    """Experiment Directory Structure
    """
    def __init__(self, expdir):
        self.expdir = expdir

    def path(self, *file):
        return os.path.join(self.expdir, *file)

    def rundir(self, runid=None):
        if runid is not None:
            runidstr = "{:0>4}".format(runid)
            rundirs = [runidstr[:-2], runidstr[-2:]] # split it (no more than 100)
            return self.path(*rundirs)
        else:
            return self.path("default")

    def top_rundirs(self, indices):
        """top rundir directories for linking
        """
        tops = ["default"]
        for i in indices:
            top = self.rundir(i).split(os.path.sep)[0]
            if top not in tops:
                tops.append(top)
        return tops

    def create_expdir(self, force=False):
        if not os.path.exists(self.expdir):
            print("create directory",self.expdir)
            os.makedirs(self.expdir)

        elif not force:
            print("error :: directory already exists: "+repr(self.expdir))
            print("     set  '--force' option to bypass this check")
            raise ValueError(self.expdir+" already exists")


class XRun(ExpDir):

    def __init__(self, model, params, expdir):
        self.model = model
        self.params = params  # XParams class
        self.expdir = expdir
 
    @classmethod
    def read(cls, expdir):
        """read from existing experiment
        """
        o = ExpDir(expdir) # dir structure
        model = GlacierModel(json.load(open(o.path("model.json"))))
        params = XParams.read(o.path("params.txt"))
        return cls(model, params, expdir)

    def setup(self, newdir=None, force=False):
        """Setup experiment directory
        """
        newdir = newdir or self.expdir

        x = ExpDir(newdir)
        x.create_expdir(force)
        print("Setup experiment directory:", newdir)
        # params txt
        print("...write params")
        self.params.write(x.path("params.txt"))
        print("...write config")
        with open(x.path("model.json"), 'w') as f:
            json.dump(self.model.config, f, indent=2, sort_keys=True)

    def link_results(self, newdir):
        assert newdir != self.expdir, 'same directories !'
        print("...link simulations results into",newdir)
        topdirs = self.top_rundirs(xrange(self.params.size))
        for top in topdirs:
            os.system("cd "+newdir+" && ln -s "+os.path.abspath(top))

    def command(self, runid=None):
        """Return command line argument to run the model
        """
        params = self.params.pset_as_dict(runid)
        rundir = self.rundir(runid=runid)
        return self.model.command(rundir, params)

    def run(self, runid=None, **kwargs):
        cmd = self.command(runid)
        rundir = self.rundir(runid)
        return run_command(cmd, rundir, **kwargs)

    def runbatch(self, array=None, background=False, submit=True, wait=False, **kwargs):
        """ Run ensemble
        """
        N = self.params.size

        # batch command
        if array is None:
            # all params by default
            array = "{}-{}".format(0, N-1) 

        # write config to expdirectory
        self.setup(force=True)  # things are up to date

        cmd = ["python", __file__, "run", self.expdir]

        #if args:
        #    cmd.append(args)
        #cmdstr = " ".join(cmd)

        if background:
            if wait:
                raise NotImplementedError("cannot wait in background mode")
            # local testing : do not use slurm
            indices = parse_slurm_array_indices(array)
            print("Run",len(indices),"out of",N,"simulations in the background")
            print(indices)
            for idx in indices:
                runcmd = cmd + ['--id',str(idx),'--background']
                os.system(" ".join(runcmd))
            return 

        # submit job to slurm (the default)
        print("Submit job array batch to SLURM")
        jobfile = os.path.join(self.expdir, "batch.sh")
        logsdir = os.path.join(self.expdir, "logs")

        runcmd = cmd + ['--id','$SLURM_ARRAY_TASK_ID']

        #os.system("rm -fr logs; mkdir -p logs") # clean logs
        if not os.path.exists(logsdir):
            os.makedirs(logsdir)

        return run_command(runcmd, self.expdir, submit=True, wait=wait,
                           array=array, jobfile=jobfile,
                output=os.path.join(logsdir, "log-%A-%a.out"),
                error=os.path.join(logsdir, "log-%A-%a.err"), **kwargs)


    # analyze ensemble
    # ----------------
    def get(self, name, runid=None):
        """Get variable 
        """
        rundir = self.rundir(runid)
        return self.model.get(rundir, name)


    def get_all(self, name):
        """Return variable for all realizations
        """
        dim = size(self.get(name, 0)) # check size of first variable
        var = np.empty((self.params.size, dim))
        var.fill(np.nan)
        for i in xrange(self.params.size):
            var[i] = self.get(name, i)
        return var.squeeze(1)


    def loglik(self, constraints, runid=None):
        """Log-like for one realization
        """
        return sum([c.logpdf( self.get(c.name, runid)) for c in constraints])


    def loglik_all(self, constraints):
        """Log-likelihood for all realizations
        """
        var = np.empty(self.params.size)
        for i in xrange(self.params.size):
            try:
                var[i] = self.loglik(constraints, i)
            except:
                var[i] = -np.inf
        return var

    
    def analyze(self, constraints, fill_array=np.nan):
        """Analyze experiment directory and return a Results objet

        Parameters
        ----------
        constraints : list of constraints
        fill_array : float or callable
            value to use instead of (skipped) array constraints (nan by default)
        """
        N = self.params.size
        state2 = np.empty((N, len(constraints)))
        state2.fill(np.nan)
        loglik2 = np.empty((N, len(constraints)))
        loglik2.fill(-np.inf)

        def reduce_array(s):
            return fill_array(s) if callable(fill_array) else fill_array

        failed = 0

        for i in xrange(N):
            try:
                state = [self.get(c.name, i) for c in constraints]
            except Exception as error:
                failed += 1
                continue

            # diagnostic per constraint
            for j, s in enumerate(state):
                loglik2[i, j] = constraints[j].logpdf(s)
                state2[i, j] = s if np.size(s) == 1 else reduce_array(s)

        print("warning :: {} out of {} simulations failed".format(failed, N))

        return Results(constraints, state2, loglik2=loglik2, params=self.params)


class Results(object):
    def __init__(self, constraints=None, state=None, 
                 loglik=None, loglik2=None, params=None, default=None):

        if loglik is None and loglik2 is not None:
            loglik = loglik2.sum(axis=1)
        self.loglik = loglik
        self.constraints = constraints   # so that names is defined

        self.state = state
        self.loglik2 = loglik2
        self.params = params
        self.default = default

        # weights
        self.loglik = loglik
        self.valid = np.isfinite(self.loglik)

    def weights(self):
        w = np.exp(self.loglik)
        return w / w.sum()

    @classmethod
    def read(cls, direc):
        x = ExpDir(direc)
        loglik = np.loadtxt(x.path("loglik.txt"))
        return cls(loglik=loglik)


    def write(self, direc):
        """write result stats and loglik to folder
        """
        print("write analysis results to",direc)
        x = ExpDir(direc)
        np.savetxt(x.path("loglik.txt"), self.loglik)

        if self.state is not None:
            with open(x.path("state.txt"), "w") as f:
                f.write(self.format(self.state))
            with open(x.path("stats.txt"), "w") as f:
                f.write(self.stats())

        if self.loglik2 is not None:
            with open(x.path("loglik.all.txt"), "w") as f:
                f.write(self.format(self.loglik2))


    @property
    def obs(self):
        return [c.mean for c in self.constraints]

    @property
    def names(self):
        return [c.name for c in self.constraints]

    def best(self):
        return self.state[np.argmax(self.loglik)]

    def mean(self):
        return self.state[self.valid].mean(axis=0)

    def std(self):
        return self.state[self.valid].std(axis=0)

    def min(self):
        return self.state[self.valid].min(axis=0)

    def max(self):
        return self.state[self.valid].max(axis=0)

    def pct(self, p):
        return np.percentile(self.state[self.valid], p, axis=0)


    def stats(self, fmt="{:.2f}", sep=" "):
        """return statistics
        """
        #def stra(a):
        #    return sep.join([fmt.format(k) for k in a]) if a is not None else "--"

        res = [
            ("obs", self.obs),
            ("best", self.best()),
            ("default", self.default),
            ("mean", self.mean()),
            ("std", self.std()),
            ("min", self.min()),
            ("p05", self.pct(5)),
            ("med", self.pct(50)),
            ("p95", self.pct(95)),
            ("max", self.max()),
        ]

        index = [nm for nm,arr in res if arr is not None]
        values = [arr for nm,arr in res if arr is not None]

        import pandas as pd
        df = pd.DataFrame(np.array(values), columns=self.names, index=index)

        return str(df) #"\n".join(lines)

    def df(self, array):
        " transform array to dataframe "
        import pandas as pd
        return pd.DataFrame(array, columns=self.names)

    def format(self, array):
        return str(self.df(array))


def decode_json(strarg):
    "to use as type in argparser"
    with open(strarg) as f:
        data = json.load(f)
    return data


class IISExp(object):
    """Handle IIS experiment
    """
    def __init__(self, initdir, constraints, iter=0, epsilon=0.05, resampling='residual'):
        self.initdir = initdir
        self.constraints = constraints
        self.iter = iter
        self.epsilon = epsilon
        self.resampling = resampling

    def goto_last_iter(self):
        while self.is_analyzed():
            self.iter += 1

    def expdir(self, iter=None):
        iter = self.iter if iter is None else iter
        return self.initdir + ('.'+str(iter)) if iter > 0 else ""

    def path(self, file, iter=None):
        return ExpDir(self.expdir(iter)).path(file)

    def xrun(self, iter=None):
        return XRun.read(self.expdir(iter))

    def is_analyzed(self, iter=None):
        return os.path.exists(self.path("loglik.txt", iter))

    def resample(self, iter):
        xrun = self.xrun(iter)
        w = np.exp(np.loadtxt(xrun.path("loglik.txt")))
        pp = xrun.params.resample(w, self.epsilon, 
                                      resampling=self.resampling)
        #pp.write(self.expdir(iter+1).path("params.txt"))  # write to current direc
        xrun.params = pp
        xrun.expdir = self.expdir(iter+1)
        return xrun

    def step(self, **kwargs):

        print("******** runiis iter={}".format(self.iter))
        assert not self.is_analyzed(), 'already analyzed'

        if self.iter == 0:
            print("*** first iteration")
            xrun = self.xun()
        else:
            print("*** resample")
            xrun = self.resample(self.iter-1)

        print("*** runbatch")
        xrun.runbatch(wait=True, **kwargs)
        print("*** analysis")
        xrun.analyze(self.constraints).write(xrun.expdir)

        # increment iterations and recursive call
        self.iter += 1

    def runiis(self, maxiter, **kwargs):
        """Iterative Importance Sampling (recursive)
        """
        while self.iter < maxiter:
            self.step(**kwargs)

        print("******** runiis terminated")


class XParser(object):
    """Helper class to build ArgumentParser with subcommand and keep clean
    """
    def __init__(self, *args, **kwargs):
        self.parser = argparse.ArgumentParser(*args, **kwargs)
        self.subparsers = self.parser.add_subparsers(dest='cmd')

    def add_setup(self):
        "setup new experiment"
        p = self.subparsers.add_parser('setup', help=self.add_setup.__doc__)

        p.add_argument('expdir', help='experiment directory')
        p.add_argument('--origin', '-o',
                       help='experiment directory to copy or update from')
        p.add_argument('--link-results', 
                       help='if --origin, also link results')

        p.add_argument('-f', '--force', action='store_true',
                       help='ignore any pre-existing directory')

        grp = p.add_argument_group('model config')
        grp.add_argument("-x", "--config-file", 
                           help="model experiment config")
        grp.add_argument("--config", default=[], type=parse_keyval, nargs='*', metavar="NAME=VALUE",
                           help="model parameter to update config file")
        grp.add_argument('-g', "--glacier", 
                           help="glacier netcdf file from observations")

        # which parameters?
        grp = p.add_argument_group('parameters')
        grp.add_argument("-p", "--params", help='params file to reuse or resample from')
        x = grp.add_mutually_exclusive_group()
        x.add_argument("--resample", action="store_true",
                         help='resample new params based on log-likelihood, need --params and --loglik')
        x.add_argument("--sample", action="store_true",
                         help='sample new params from prior')

        grp.add_argument("-N", "--size", type=int, help="ensemble size")
        grp.add_argument("--seed", type=int, help="random state")

        grp.add_argument("--prior-file", 
                           help="prior config file, required if --sample")
        grp.add_argument("--prior-params", nargs='*', default=[], type=PriorParam.parse,
                           help="command-line prior params, to update config file")
        grp.add_argument("--loglik", 
                           help="log-likelihood file, required if --resample")
        grp.add_argument("--epsilon", default=None, type=float, 
                         help="resample: loglik weight + jitter")
        grp.add_argument("--resampling-method", default="residual", 
                         help="resample: loglik weight + jitter")
        grp.add_argument("--sampling-method", default="lhs", 
                         choices=["lhs", "montecarlo"], help="sample: lhe")

        return p

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

    def add_runbatch(self):
        "run ensemble"
        p = self.subparsers.add_parser("runbatch", 
                                   help=self.add_runbatch.__doc__)
        p.add_argument("expdir", help="experiment directory (need to setup first)")

        #p.add_argument("--args", help="pass on to glacier")
        p.add_argument("--background", action="store_true", 
                          help="run in background instead of submitting to slurm queue")
        p.add_argument("--array",'-a', help="slurm sbatch --array")
        p.add_argument("--wait", action="store_true")
        self.add_slurm_group(p)
        return p

    def add_loglik(self):
        p =  self.subparsers.add_parser("loglik", 
                               help="return log-likelihood for one run")
        p.add_argument("expdir", help="experiment directory (need to setup first)")
        p.add_argument("--id", type=int, help='specify only on run')
        p.add_argument("-l", "--constraints-file", 
                       help="constraints to compute likelihood")
        return p

    def add_constraints_group(self, subp):
        grp = subp.add_argument_group("obs constraints")
        grp.add_argument("--obs-file", help="obs constraints config file")
        grp.add_argument("--obs", nargs='*', default=[], help="list of obs constraints")

    def add_analysis(self):
        """analysis for the full ensemble: state, loglik, etc...
        """
        subp = self.subparsers.add_parser("analysis", help=self.add_analysis.__doc__)
        subp.add_argument("expdir", help="experiment directory (need to setup first)")
        self.add_constraints_group(subp)
        subp.add_argument('-f', '--force', action='store_true',
                       help='force analysis even if loglik.txt already present')
        return subp

    def add_iis(self):
        """run a number of iterations following IIS methodology
        """
        # perform IIS optimization
        subp = self.subparsers.add_parser("iis", parents=[parent], 
                                   help=self.add_iis.__doc__)
        subp.add_argument("expdir", help="experiment directory (need to setup first)")
        self.add_constraints_group(subp)
        subp.add_argument("-n", "--maxiter", type=int, required=True, 
                          help="max number of iterations to reach")
        subp.add_argument("--start", type=int, default=0,
                          help="start from iteration (default=0), note: previous iter must have loglik.txt file")
        subp.add_argument("--restart", action='store_true', 
                          help="automatically find start iteration")
        subp.add_argument("--epsilon", default=None, type=float, 
                help="loglik weight + jitter")
        return subp

    def parse_args(self, *args, **kwargs):
        return self.parser.parse_args(*args, **kwargs)


def get_constraints(args, getobs):
    like = Likelihood.read(args.obs_file, getobs)
    constraints = [parse_constraint(cstring, getobs=getobs) 
                   for cstring in args.obs]
    like.update(constraints)
    return like.constraints


def main(argv=None):

    parser = XParser(description=__doc__)
    parser.add_setup()
    parser.add_run()
    parser.add_runbatch()
    #parser.add_loglik()
    parser.add_analysis()
    #parser.add_iis()

    args = parser.parse_args(argv)

    if args.cmd == "setup":
        x = ExpDir(args.expdir)
        x.create_expdir(args.force)  # for resampling

        # define default files from origin directory
        if args.origin:
            o = XRun.read(args.origin)
            if not args.config_file and os.path.exists(o.path("model.json")): 
                args.config_file = o.path("model.json")
            if not args.params and os.path.exists(o.path("params.txt")): 
                args.params = o.path("params.txt")
            # indicate origin directory
            with open(x.path("origin"), 'w') as f:
                f.write(args.origin)

            if args.link_results:
                o.link_results(args.expdir)

        # glacier model
        if args.config_file:
            config = json.load(open(args.config_file))
        else:
            config = {}
        if args.glacier:
            config["in_file"] = args.glacier  # temporary fix
        model = GlacierModel(config)
        model.update(dict(args.config))

        # parameters
        if args.sample:
            # sample from prior parameters
            assert args.size is not None, "need to provide --size for sampling"
            prior = PriorParams.read(args.prior_file)
            prior.update(args.prior_params)
            params = prior.sample(args.size, seed=args.seed, method=args.sampling_method)

        elif args.resample:
            assert args.origin is not None, 'need to pass --origin when --resample'
            # sample from prior parameters
            print("Resample",args.params,"with",args.origin,"loglik into",args.expdir)
            prev = XParams.read(args.params)
            res = Results.read(args.origin)
            params = prev.resample(np.exp(res.loglik), args.epsilon, 
                                  size=args.size, seed=args.seed,
                                  method=args.resampling_method)
        else:
            # simply read !
            params = XParams.read(args.params)

        xrun = XRun(model, params, args.expdir)
        xrun.setup(force=True)  # already created anyway

    elif args.cmd == "run":

        xrun = XRun.read(args.expdir)
        xrun.run(runid=args.id, dry_run=args.dry_run, background=args.background)

    elif args.cmd == "runbatch":

        xrun = XRun.read(args.expdir)
        xrun.runbatch(array=args.array, background=args.background, 
                      qos=args.qos, job_name=args.job_name, account=args.account, time=args.time, wait=args.wait)


    elif args.cmd == "analysis":

        # model config & params already present
        print("analysis of experiment", args.expdir)
        xrun = XRun.read(args.expdir)

        if os.path.exists(xrun.path("loglik.txt")) and not args.force:
            raise ValueError("analysis already performed, use --force to overwrite")

        # define constraints
        constraints = get_constraints(args, xrun.model.getobs)

        # analyze
        results = xrun.analyze(constraints)
        results.write(args.expdir)


    elif args.cmd == "iis":

        constraints = get_constraints(args, xrun.model.getobs)

        iis = IISExp(args.expdir, constraints, iter=args.start, epsilon=args.epsilon, 
                     resampling=args.resampling_method)

        if args.restart:
            iis.goto_last_iter()
        iis.runiis(args.maxiter)

    else:
        raise NotImplementedError("subcommand not yet implemented: "+args.cmd)


if __name__ == '__main__':
    main()
