#!/usr/bin/env python2.7
"""Process configuration file
"""
from __future__ import print_function, absolute_import
import numpy as np
import json
import os
import sys
from collections import OrderedDict as odict

from simtools.modelrun import autoset_params, maybe_transform_param, run_background, run_foreground
from simtools.modelstate import read_model

# directory structure
glacierexe = "glacier"
glaciersdir = "glaciers"
experimentsdir = "experiments"
configfile = "config.json"

class GlobalConfig(object):
    def __init__(self, data):
        self.data = data

    @classmethod
    def read(cls, file=configfile):
        import json
        return cls(json.load(open(file)))

    def get_expconfig(self, name):
        """Return ExperimentConfig class
        """
        expnames = [exp["name"] for exp in self.data["experiments"]]
        try:
            i = expnames.index(name)
        except:
            print("available experiments: ", expnames)
            raise
        return ExperimentConfig(self.data["experiments"][i])


class ExperimentConfig(object):
    def __init__(self, data, expdir=None):
        self.data = data
        if expdir is None:
            expdir = os.path.join(experimentsdir, self.name)
        self.expdir = expdir

    def glaciernc(self, glacier, runid=None):
        return os.path.join(glaciersdir, glacier+".nc")

    def rundir(self, runid=None):
        if runid is not None:
            return os.path.join(self.expdir, "{:0>5}".format(runid))
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
        return self.prior["size"]

    def genparams_args(self, out=None):
        """Generate command-line arguments for genparams script from json dict
        """
        prior = self.prior
        cmd = ["--params"]
        for p in prior["params"]:
            lo, up = p["range"]
            cmd.append("{}=uniform?{},{}".format(p["name"],lo, up))

        cmd.append("--mode={}".format(prior["sampling"]))
        cmd.append("--size={}".format(self.get_size()))
        if prior["seed"] is not None:
            cmd.append("--seed={}".format(prior["seed"]))

        if out is not None:
            cmd.append("--out="+out)
        return " ".join(cmd)

    
    def genparams(self, log=None):
        """Return prior parameters
        """
        if log is None:
            log = os.path.join(self.expdir, "params.cmd")
        args = self.genparams_args(self.paramsfile)
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


    def glacierargs(self, glacier, runid=None, outdir=None, cmd_extra=""):
        """Return glacier executable and glacier arguments
        """
        # first create a dictionary of parameters
        params = odict()
        netcdf = self.glaciernc(glacier, runid)
        
        # default arguments
        for k in sorted(self.default.keys()):
            params[k] = self.default[k]

        # data-dependent parameters
        tauc_max, uq, h0 = autoset_params(netcdf)
        params["dynamics%tauc_max"] = tauc_max
        params["dynamics%Uq"] = uq
        params["dynamics%H0"] = h0

        # update arguments from file
        if runid is not None:
            pnames, pmatrix = self.getparams()
            pvalues = pmatrix[runid]
            for k, v in zip(pnames, pvalues):
                params[k] = v

        if outdir is None:
            outdir = self.rundir(runid=runid)

        # make command line argument for glacier executable
        cmd = ["--in_file", netcdf, "--out_dir",outdir]
        for k in params:
            name, value = maybe_transform_param(k, params[k])
            cmd.append("--"+name)
            cmd.append(str(value))

        cmdstr = " ".join(cmd) + (" " + cmd_extra if cmd_extra else "")
        return glacierexe, cmdstr


def parse_slurm_array_indices(a):
    indices = []
    for i in a.split(","):
        if '-' in i:
            if ':' in i:
                i, step = i.split(':')
                step = int(step)
            else:
                step = 1
            start, stop = i.split('-')
            start = int(start)
            stop = int(stop) + 1  # last index is ignored in python
            indices.extend(range(start, stop, step))
        else:
            indices.append(int(i))
    return indices


def make_jobfile_slurm(command, queue, jobname, account, output, error):
    return """#!/bin/bash

#SBATCH --qos={queue}
#SBATCH --job-name={jobname}
#SBATCH --account={account}
#SBATCH --output={output}
#SBATCH --error={error}

echo
echo SLURM JOB
echo ---------
echo "SLURM_JOBID $SLURM_JOBID"
echo "SLURM_ARRAY_JOB_ID $SLURM_ARRAY_JOB_ID"
echo "SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID"
cmd="{command}"
echo $cmd
eval $cmd""".format(**locals())


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='cmd')

    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--config", default=configfile, 
                        help="experiment config (default=%(default)s)")
    parent.add_argument("--experiment", required=True, #default="steadystate", 
                        help="experiment name") # (default=%(default)s)")
    parent.add_argument("--expdir", 
                        help="experiment directory") # (default=%(default)s)")

    subparsers.add_parser('genparams', parents=[parent], 
                               help="generate ensemble")
    subp = subparsers.add_parser('get', parents=[parent], 
                               help="get config field")
    subp.add_argument('field')

    # model run
    runpars = argparse.ArgumentParser(add_help=False)
    runpars.add_argument("glacier", help="glacier name")
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

    # cost function
    subp = subparsers.add_parser("costfunction", parents=[parent], 
                               help="extract cost function")

    args = parser.parse_args(argv)

    cfg = GlobalConfig.read(args.config)
    expcfg = cfg.get_expconfig(args.experiment)
    if args.expdir:
        expcfg.expdir = args.expdir

    if args.cmd == "genparams":
        # generate parameters if not present
        if not os.path.exists(expcfg.paramsfile):
            expcfg.genparams()
        else:
            print(expcfg.paramsfile, "already exists, do nothing")

    elif args.cmd == "get":
        print(getattr(expcfg, args.field))

    elif args.cmd == "run":

        outdir = expcfg.rundir(args.id)
        exe, cmdstr = expcfg.glacierargs(args.glacier, runid=args.id, cmd_extra=args.args, outdir=outdir)

        print(exe, cmdstr)
        
        if not args.dry_run:

            if not os.path.exists(outdir):
                os.makedirs(outdir)

            logfile = os.path.join(outdir, "glacier.log")

            if args.background:
                run_background(exe, cmd_args=cmdstr, logfile=logfile)

            else:
                ret = run_foreground(exe, cmd_args=cmdstr, logfile=logfile)
                # check return results
                if ret != 0:
                    raise RuntimeError("Error when running the model")
                if not os.path.exists("simu_ok"):
                    raise RuntimeError("simu_ok not written: error when running the model")

                print("Simulation successful")

    elif args.cmd == "runbatch":

        # check out ensemble size
        pnames, pmatrix = expcfg.getparams()
        N = len(pmatrix)
        assert N == expcfg.get_size(), "mistmatch between experiment specs and "+expcfg.paramsfile

        # batch command
        if args.array is None:
            # all params by default
            args.array = "{}-{}".format(0, N-1) 

        cmd = ["python", __file__, "run", args.glacier, 
                "--experiment", args.experiment, 
                "--expdir", expcfg.expdir,
                "--id", "{id}"]
        if args.args:
            cmd.append(args.args)
        cmdstr = " ".join(cmd)

        if args.background:
            # local testing : do not use slurm
            indices = parse_slurm_array_indices(args.array)
            print("Run",len(indices),"out of",N,"simulations in the background")
            print(indices)
            for idx in indices:
                os.system(cmdstr.format(id=idx)+' --background')
            return 

        # submit job to slurm (the default)
        print("Submit job array batch to SLURM")
        jobfile = os.path.join(expcfg.expdir, "batch.sh")
        logsdir = os.path.join(expcfg.expdir, "logs")
        #os.system("rm -fr logs; mkdir -p logs") # clean logs
        if not os.path.exists(logsdir):
            os.makedirs(logsdir)

        jobtxt = make_jobfile_slurm(cmdstr.format(id="$SLURM_ARRAY_TASK_ID"), 
                queue="short", 
                jobname=__file__, 
                account="megarun",
                output=os.path.join(logsdir, "log-%A-%a.out"),
                error=os.path.join(logsdir, "log-%A-%a.err"),
                )
        
        with open(jobfile, "w") as f:
            f.write(jobtxt)

        batchcmd = ["sbatch", "--array", args.array, jobfile]
        cmd = " ".join(batchcmd)

        os.system("echo "+cmd)
        os.system("echo "+cmd+" >> "+os.path.join(expcfg.expdir, "batch.submit"))
        os.system("eval "+cmd)


    elif args.cmd == "costfunction":
        pass

    else:
        raise NotImplementedError("subcommand not yet implemented: "+args.cmd)


if __name__ == '__main__':
    main()
