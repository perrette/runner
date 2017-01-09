#!/usr/bin/env python2.7
"""Process configuration file
"""
from __future__ import print_function, absolute_import
import numpy as np
import json
import copy
import os
import sys
import subprocess
from collections import OrderedDict as odict

from glaciermodel import glacierargs, read_model
from simtools.modelrun import (run_background, run_foreground, 
                               make_jobfile_slurm, 
                               parse_slurm_array_indices, wait_for_jobid)
from simtools.resample import read_params

from simtools.costfunction import Normal, RMS
import netCDF4 as nc

# default directory structure
glaciersdir = "glaciers"
experimentsdir = "experiments"

def nans(N):
    a = np.empty(N)
    a.fill(np.nan)
    return a


class GlobalConfig(object):
    def __init__(self, data):
        self.data = data

    @classmethod
    def read(cls, file):
        import json
        return cls(json.load(open(file)))

    def get_expconfig(self, name, glacier, expdir=None):
        """Return ExperimentConfig class
        """
        expnames = [exp["name"] for exp in self.data["experiments"]]
        try:
            i = expnames.index(name)
        except:
            print("available experiments: ", expnames)
            raise
        return ExperimentConfig(self.data["experiments"][i], glacier, expdir)


class ExperimentConfig(object):
    def __init__(self, data, glacier, expdir=None):
        self.glacier = glacier
        self.data = data
        if expdir is None:
            expdir = os.path.join(experimentsdir, self.glacier, self.name)
        self.expdir = expdir

    def glaciernc(self, runid=None):
        return os.path.join(glaciersdir, self.glacier+".nc")

    def rundir(self, runid=None):
        if runid is not None:
            runidstr = "{:0>4}".format(runid)
            rundirs = [runidstr[:-2], runidstr[-2:]] # split it (no more than 100)
            return os.path.join(self.expdir, *rundirs)
        else:
            return os.path.join(self.expdir, "default")

    @property 
    def paramsfile(self):
        return os.path.join(self.expdir, "params.txt")

    @property 
    def name(self):
        return self.data["name"]

    @property 
    def constraints(self):
        return self.data["constraints"]

    @property 
    def prior(self):
        return self.data["prior"]

    @property 
    def default(self):
        return self.data["default"]

    def get_size(self):
        pnames, pmatrix = read_params(self.paramsfile)
        size = len(pmatrix)
        return size

    def genparams_args(self, out=None, size=None):
        """Generate command-line arguments for genparams script from json dict
        """
        prior = self.prior
        cmd = ["--params"]
        for p in prior["params"]:
            lo, up = p["range"]
            cmd.append("{}=uniform?{},{}".format(p["name"],lo, up-lo))

        cmd.append("--mode={}".format(prior["sampling"]))
        cmd.append("--size={}".format(size or self.prior["size"]))
        if prior["seed"] is not None:
            cmd.append("--seed={}".format(prior["seed"]))

        if out is not None:
            cmd.append("--out="+out)
        return " ".join(cmd)

    
    def genparams(self, log=None, size=None):
        """Return prior parameters
        """
        if log is None:
            log = os.path.join(self.expdir, "params.cmd")
        args = self.genparams_args(self.paramsfile, size=size)
        if (os.path.dirname(self.paramsfile) == self.expdir 
                and not os.path.exists(self.expdir)):
            os.makedirs(self.expdir) # make experiment directory is not present
        cmd = "python -m simtools.genparams "+args
        print(cmd)
        os.system(cmd)
        os.system("echo "+ cmd + " > "+log)


    def getparams(self, pfile=None):
        """Return experiment parameters (read from disk)
        """
        pfile = pfile or self.paramsfile
        pnames = open(pfile).readline().split()
        pvalues = np.loadtxt(pfile, skiprows=1)  
        return pnames, pvalues


    def glacierargs(self, runid=None, outdir=None, cmd_extra=""):
        """Return glacier executable and glacier arguments
        """
        netcdf = self.glaciernc(runid)

        # create a dictionary of parameters
        params = self.default.copy()

        # update arguments from file
        if runid is not None:
            pnames, pmatrix = self.getparams()
            pvalues = pmatrix[runid]
            for k, v in zip(pnames, pvalues):
                params[k] = v

        if outdir is None:
            outdir = self.rundir(runid=runid)

        if cmd_extra:
            args = cmd_extra.split()
        else:
            args = []

        cmd = glacierargs(netcdf, outdir, *args, **params)
        glacierexe = cmd[0]
        cmdstr = " ".join(cmd[1:])
        return glacierexe, cmdstr

    def get_analysis(self, constraintscfg=None, obsnc=None):
        return Analysis(self, constraintscfg, obsnc)


    def run(self, runid=None, args=None, dry_run=None, background=None):

        outdir = self.rundir(runid)
        exe, cmdstr = self.glacierargs(runid=runid, cmd_extra=args, outdir=outdir)
        print(exe, cmdstr)
        
        if not dry_run:

            if not os.path.exists(outdir):
                os.makedirs(outdir)

            logfile = os.path.join(outdir, "glacier.log")

            if background:
                run_background(exe, cmd_args=cmdstr, logfile=logfile)

            else:
                ret = run_foreground(exe, cmd_args=cmdstr, logfile=logfile)


    def runbatch(self, array=None, background=None, args=None, wait=False, walltime="0:2:0"):
        """ Run ensemble
        """
        # check out ensemble size
        pnames, pmatrix = self.getparams()
        N = len(pmatrix)

        # batch command
        if array is None:
            # all params by default
            array = "{}-{}".format(0, N-1) 

        cmd = ["python", __file__, "run", "--glacier", self.glacier, 
                "--experiment", self.name, 
                "--expdir", self.expdir,
                "--id", "{id}"]

        if args:
            cmd.append(args)
        cmdstr = " ".join(cmd)

        if background:
            if wait:
                raise NotImplementedError("cannot wait in background mode")
            # local testing : do not use slurm
            indices = parse_slurm_array_indices(array)
            print("Run",len(indices),"out of",N,"simulations in the background")
            print(indices)
            for idx in indices:
                os.system(cmdstr.format(id=idx)+' --background')
            return 

        # submit job to slurm (the default)
        print("Submit job array batch to SLURM")
        jobfile = os.path.join(self.expdir, "batch.sh")
        logsdir = os.path.join(self.expdir, "logs")
        #os.system("rm -fr logs; mkdir -p logs") # clean logs
        if not os.path.exists(logsdir):
            os.makedirs(logsdir)

        jobtxt = make_jobfile_slurm(cmdstr.format(id="$SLURM_ARRAY_TASK_ID"), 
                queue="short", 
                jobname=__file__, 
                account="megarun",
                output=os.path.join(logsdir, "log-%A-%a.out"),
                error=os.path.join(logsdir, "log-%A-%a.err"),
                time=walltime,
                )
        
        with open(jobfile, "w") as f:
            f.write(jobtxt)

        batchcmd = ["sbatch", "--array", array, jobfile]
        cmd = " ".join(batchcmd)

        os.system("echo "+cmd)
        os.system("echo "+cmd+" >> "+os.path.join(self.expdir, "batch.submit"))
        output = subprocess.check_output("eval "+cmd, shell=True)
        arrayjobid = output.split()[-1]

        if wait:
            wait_for_jobid(arrayjobid)

        return arrayjobid

class Analysis(object):
    """Compute log-likelood
    """
    def __init__(self, cfg, constraintscfg=None, obsnc=None):
        """Initialize from config, optionally using different constraints
        """
        self.cfg = cfg   # experiment config
        self.obsnc = obsnc or cfg.glaciernc()
        constraintscfg = constraintscfg or cfg.constraints
        self.constraints = self._get_constraints(constraintscfg, self.obsnc)
        self.names = [c["name"] for c in constraintscfg]

    @staticmethod
    def _get_constraints(constraintscfg, obsnc):
        """Return a list of Constraints based on config
        """
        constraints = []
        for c in constraintscfg:
            obs = read_model(obsnc, c["name"])
            assert np.isscalar(obs)
            if 'std_as_fraction' in c:
                std = obs*c['std_as_fraction']
            else:
                std = c['std']
            constraints.append(
                Normal(c["name"], obs, std, desc=c["desc"])
            )
        return constraints

    @staticmethod
    def _loglik(ds, name, c, runid=None):
        """Return log-likelihood for one constraint
        """
        state = read_model(ds, name)
        loglik = c.logpdf(state)
        return loglik

    def loglik(self, runid=None, netcdf=None, verbose=False):
        """Return log-likelihood given all constraints
        """
        if netcdf is None:
            outdir = self.cfg.rundir(runid)
            if not os.path.exists(os.path.join(outdir, 'simu_ok')):
                if verbose: print("simulation failed:", runid)
                return -np.inf
            netcdf = os.path.join(outdir,'restart.nc')

        with nc.Dataset(netcdf) as ds:
            loglik = 0
            for name, c in zip(self.names, self.constraints):
                loglik += self._loglik(ds, name, c, runid)
        return loglik

    def loglik_ensemble(self, indices=None):
        if indices is None:
            N = self.cfg.get_size()
            indices = np.arange(N)

        loglik = np.empty(indices.size)
        loglik.fill(-np.inf)

        for i in indices:
            loglik[i] = self.loglik(i)
        return loglik

    def state(self, runid=None, netcdf=None):
        """Return model state as a list of states
        """
        if not netcdf:
            outdir = self.cfg.rundir(runid)
            if not os.path.exists(os.path.join(outdir, 'simu_ok')):
                raise ValueError("Simulation failed: "+str(runid) 
                                 if runid is not None else os.path.join(outdir))

            netcdf = os.path.join(outdir,'restart.nc')

        with nc.Dataset(netcdf) as ds:
            state = []
            for name in self.names:
                state.append( read_model(ds, name) )
        return state

    def state_ensemble(self, indices=None):
        """Return a diagnostic matrix of ensemble state, scalar variables only
        """
        if indices is None:
            N = self.cfg.get_size()
            indices = np.arange(N)

        state = np.empty((indices.size, len(self.constraints)))
        state.fill(np.nan)

        for i in indices:
            try:
                statevars = self.state(i)
            except:
                continue
            for j, stat in enumerate(statevars):
                if np.ndim(stat) == 0:   # scalar only
                    state[i, j] = stat
        return state

    def getloglikfile(self, anadir=None):
        anadir = anadir or self.cfg.expdir
        return os.path.join(anadir, "loglik.txt")

    def write_analysis(self, anadir=None):
        """Perform analysis, given constraints

        + loglik.txt
        + state.txt
        """
        print("loglik of ensemble:", self.cfg.expdir)
        if anadir is not None:
            print("analysis written to",anadir)
        anadir = anadir or self.cfg.expdir
        if not os.path.exists(anadir):
            os.makedirs(anadir)
        loglik = self.loglik_ensemble()
        loglikfile = self.getloglikfile(anadir)
        print("write loglik to",loglikfile)
        np.savetxt(loglikfile, loglik)

        # constraints file
        constraintsfile = os.path.join(anadir, "constraints.json")
        print("copy constraints cfg to",constraintsfile)
        with open(constraintsfile, "w") as f:
            json.dump(self.cfg.data["constraints"], f, indent=2)

        # state data
        print("state of ensemble:", anadir)
        # ensemble state
        header = ", ".join(self.names)
        state = self.state_ensemble()
        footer = "      "+", ".join(self.names)
        fmt = "{:.2f}"
        if os.path.exists(self.obsnc):
            stateobs = [v if np.ndim(v) == 0 else np.nan 
                        for v in self.state(netcdf=self.obsnc)]
            footer += "\nobs: "+", ".join(fmt.format(v) for v in stateobs)
        else:
            print("warning: obs file", self.obs, "not found")
        try:
            statedef = [v if np.ndim(v) == 0 else np.nan for v in self.state()]
            footer += "\ndef: "+", ".join(fmt.format(v) for v in statedef)
        except Exception as error:
            print("warning: default run:", error.message)

        # most likely
        i = np.argmax(loglik)
        footer += "\nbest: "+", ".join([fmt.format(v) for v in state[i]])
        state2 = state[np.any(~np.isnan(state), axis=1)] # no nans
        footer += "\n -- "
        footer += "\nmean: "+", ".join([fmt.format(v) for v in np.nanmean(state,axis=0)])
        footer += "\nstd: "+", ".join([fmt.format(v) for v in np.nanstd(state,axis=0)])
        footer += "\n -- "
        footer += "\nmin: "+", ".join([fmt.format(v) for v in np.nanmin(state,axis=0)])
        footer += "\np05: "+", ".join([fmt.format(v) for v in np.percentile(state2,5,axis=0)])
        footer += "\nmed: "+", ".join([fmt.format(v) for v in np.median(state2,axis=0)])
        footer += "\np95: "+", ".join([fmt.format(v) for v in np.percentile(state2,95,axis=0)])
        footer += "\nmax: "+", ".join([fmt.format(v) for v in np.nanmax(state,axis=0)])
        footer += "\n -- "

        statefile = os.path.join(anadir, "state.txt")
        print("write state to",statefile)
        np.savetxt(statefile, state, header=header, footer=footer, fmt="%.2f")


def decode_json(strarg):
    "to use as type in argparser"
    with open(strarg) as f:
        data = json.load(f)
    return data


def resample_exp_params(oldexpdir, newexpdir, epsilon):
    """Resample parameters from oldexpdir to newexpdir
    """
    if not os.path.exists(newexpdir):
        os.makedirs(newexpdir)

    cmd = "./scripts/resample -w {oldexpdir}/loglik.txt --log --epsilon {expsilon} --jitter -p {oldexpdir}/params.txt  > {expdir}/params.txt".format(oldexpdir=oldexpdir, expdir=newexpdir, expsilon=epsilon)
    print(cmd)
    assert os.system(cmd) == 0, "failed to resample parameters"


class IISExp(object):
    """Handle IIS experiment
    """
    def __init__(self, basecfg, iter=0, epsilon=0.05):
        self.basecfg = basecfg
        self.iter = iter
        self.epsilon = epsilon


    def goto_last_iter(self):
        while os.path.exists(self.getcfg().paramsfile):
            self.iter += 1

    def getcfg(self, iter=None):
        if iter is None:
            iter = self.iter
        if iter == 0:
            return self.basecfg
        cfg = copy.copy(self.basecfg)
        #cfg.expdir = os.path.join(self.iisdir, "iis.{:0>2}".format(iter))
        cfg.expdir = self.basecfg.expdir+'.'+str(iter)
        return cfg

    def runiis(self, niter, **kwargs):
        """Iterative Importance Sampling (recursive)
        """
        if niter == 0:
            print("******** runiis terminated")
            return 

        cfg = self.getcfg()
        if not os.path.exists(cfg.paramsfile):
            if self.iter == 0:
                raise ValueError("No parameter file: "+cfg.paramsfile)
            else:
                oldcfg = self.getcfg(self.iter - 1)
                try:
                    print("*** resample")
                    resample_exp_params(oldcfg.expdir, cfg.expdir, self.epsilon)
                except Exception as error:
                    print("Previous analysis not performed?")
                    raise


        print("******** runiis iter={}".format(self.iter))
        print("*** runbatch")
        cfg.runbatch(wait=True, **kwargs)
        print("*** analysis")
        cfg.get_analysis().write_analysis()

        # increment iterations and recursive call
        self.iter += 1
        return self.runiis(niter-1, **kwargs)



def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='cmd')

    parent = argparse.ArgumentParser(add_help=False)
    grp = parent.add_argument_group("root directory structure")
    parent.add_argument("--config", default="config.json", 
                        help="experiment config (default=%(default)s)")
    parent.add_argument("--experiment", default="steadystate", #default="steadystate", 
                        help="experiment name") # (default=%(default)s)")
    parent.add_argument("--expdir", help="specifiy experiment directory")
    parent.add_argument("--glacier", default="daugaard-jensen", help="glacier name")

    # sample prior params
    subp = subparsers.add_parser('genparams', parents=[parent], 
                               help="generate ensemble")
    subp.add_argument("--size", type=int,
                        help="ensemble size (if different from config.json)")

    # resample params
    subp = subparsers.add_parser('resample', parents=[parent], 
                               help="generate ensemble")
    subp.add_argument("--size", type=int,
                        help="ensemble size (if different from previous)")
    subp.add_argument("--epsilon", default=0.05, type=float, 
            help="loglik weight + jitter")
    subp.add_argument("--oldexpdir", required=True, 
            help="old experiment directory")

    # get expconfig fields
    subp = subparsers.add_parser('get', parents=[parent], 
                               help="get config field")
    subp.add_argument('field')

    # model run
    runpars = argparse.ArgumentParser(add_help=False)
    runpars.add_argument("--args", help="pass on to glacier")

    subp = subparsers.add_parser("run", parents=[parent, runpars], 
                               help="run ensemble")
    subp.add_argument("--id", type=int, help="run id")
    subp.add_argument("--dry-run", action="store_true",
                      help="do not execute, simply print the command")
    subp.add_argument("--background", action="store_true",
                        help="run in the background, do not check result")

    # ensemble run
    subp = subparsers.add_parser("runbatch", parents=[parent, runpars], 
                               help="run ensemble")
    subp.add_argument("--background", action="store_true", 
                      help="run in background instead of submitting to slurm queue")
    subp.add_argument("--array",'-a', help="slurm sbatch --array")
    subp.add_argument("--job-script", default="job.sh", 
            help="job script to run the model with slurm --array (default=%(default)s)")

    # loglikelihood for one run
    subp = subparsers.add_parser("loglik", parents=[parent], 
                               help="return log-likelihood for one run")
    subp.add_argument("--id", type=int, help='specify only on run')

    # analysis for the full ensemble: state, loglik, etc...
    subp = subparsers.add_parser("analysis", parents=[parent], 
                               help="write state and log-likelihood for the ensemble")
    subp.add_argument('--anadir', help='directory to write loglik.txt and state.txt, if different from expdir')
    subp.add_argument('--constraints', type=decode_json, help='json file to read the constraints from, if different from config.json')


    # perform IIS optimization
    subp = subparsers.add_parser("iis", parents=[parent], 
                               help="run a number of iterations following IIS methodology")
    subp.add_argument("-n", "--niter", type=int, required=True, 
                      help="number of iterations to perform")
    subp.add_argument("--start", type=int, default=0,
                      help="start from iteration (default=0), note: previous iter must have loglik.txt file")
    subp.add_argument("--restart", action='store_true', 
                      help="automatically find start iteration")
    subp.add_argument("--epsilon", default=0.05, type=float, 
            help="loglik weight + jitter")
    #subp.add_argument("-n", "--size", type=int,
    #                  help="sample size, if different from original experiment")
    #subp.add_argument("--oldexpdir", 
    #                  help="old experiment directory, from which to start from if no params.txt is present")

    args = parser.parse_args(argv)

    cfg = GlobalConfig.read(args.config)
    expcfg = cfg.get_expconfig(args.experiment, args.glacier, args.expdir)

    if args.cmd == "genparams":
        # generate parameters if not present
        if not os.path.exists(expcfg.paramsfile):
            expcfg.genparams(size=args.size)
        else:
            print(expcfg.paramsfile, "already exists, do nothing")

    elif args.cmd == "resample":

        resample_exp_params(args.oldexpdir, args.expdir, args.epsilon)

    elif args.cmd == "get":
        print(getattr(expcfg, args.field))

    elif args.cmd == "run":
        expcfg.run(args.id, args.args, args.dry_run, args.background)

    elif args.cmd == "runbatch":
        expcfg.runbatch(args.array, args.background, args.args)

    elif args.cmd == "loglik":
        # for checking: one simulation
        print("loglik of simu:", expcfg.rundir(args.id))
        ana = expcfg.get_analysis()
        loglik = ana.loglik(args.id)
        print(loglik)
        sys.exit(0)


    elif args.cmd == "analysis":
        # loglikelihood
        ana = expcfg.get_analysis(constraintscfg=args.constraints)
        ana.write_analysis(args.anadir)


    elif args.cmd == "iis":
        iis = IISExp(expcfg, iter=args.start, epsilon=args.epsilon)
        if args.restart:
            iis.goto_last_iter()
        iis.runiis(args.niter)

    else:
        raise NotImplementedError("subcommand not yet implemented: "+args.cmd)


if __name__ == '__main__':
    main()
