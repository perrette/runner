"""Experiment run
"""
from __future__ import print_function, absolute_import
import argparse
import numpy as np
import json
import copy
import os
import sys
import subprocess

from simtools.model import Param, Model
from simtools.submit import submit_job
from simtools.xparams import XParams

DIGIT = 4  # number of digits for output folders
EMPTYCONFIG = "empty.json" # todo path


# Ensemble Xperiment
# ==================

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

# Environment I/O to communicate with model
# =========================================

# prefix for environ variables
ENVPREFIX = "SIMTOOLS_"

def getenv(name, *failobj):
    " for runner to set environment variables" 
    return os.environ.get(ENVPREFIX+name.upper(), *failobj)

def makenv(context, env=None):
    env = env or {}
    for k in context:
        if context[k] is not None:
            env[ENVPREFIX+k.upper()] = context[k]
    return env



def _create_dirtree(a,chunksize=2):
    """create a directory tree from a single, long name

    e.g. "12345" --> ["1", "23", "45"]
    """
    b = a[::-1]  # reverse
    i = 0
    l = []
    while i < len(b):
        l.append(b[i:i+chunksize])
        i += chunksize
    return [e[::-1] for e in l[::-1]]

class XDir(object):
    """Experiment Directory Structure
    """
    def __init__(self, expdir, digit=DIGIT):
        self.expdir = expdir
        self.digit = digit

    def path(self, *file):
        return os.path.join(self.expdir, *file)

    def runtag(self, runid=None):
        "provide a tag with same length accross the ensemble"
        digit = self.digit or len(str(self.size()-1))
        fmt = "{:0>"+str(self.digit)+"}"
        return fmt.format(runid) if runid is not None else "default"

    def rundir(self, runid=None):
        if runid is not None:
            runtag = self.runtag(runid)
            rundirs = _create_dirtree(runtag)
            return self.path(*rundirs)
        else:
            return self.path("default")

    def logdir(self, runid=None):
        return self.path("logs")

    def logout(self, runid=None):
        return os.path.join(self.logdir(runid), 'log-{runid}.out').format(runid)

    def logerr(self, runid=None):
        return os.path.join(self.logdir(runid), 'log-{runid}.err').format(runid)

    def statefile(self, runid=None):
        """state variable name
        """
        runtag = self.runtag(runid)
        return self.path(self.expdir, "state.{}".format(runtag))

    def create_expdir(self, force=False):
        if not os.path.exists(self.expdir):
            print("create directory",self.expdir)
            os.makedirs(self.expdir)

        elif not force:
            print("error :: directory already exists: "+repr(self.expdir))
            print("     set  '--force' option to bypass this check")
            raise ValueError(self.expdir+" already exists")

    def top_rundirs(self, indices):
        """top rundir directories for linking
        """
        tops = ["default"]
        for i in indices:
            top = self.rundir(i).split(os.path.sep)[0]
            if top not in tops:
                tops.append(top)
        return tops

    def link_results(self, orig):
        """Link results from a previous expdir
        """
        assert orig != self.expdir, 'same directories !'
        print("...link simulations results from",orig)
        x = XDir(orig)
        topdirs = x.top_rundirs(xrange(self.size()))
        for top in topdirs:
            os.system("cd "+self.expdir+" && ln -s "+os.path.abspath(top))

    def size(self):
        return XParams.read(self.path("params.txt")).size


class XRun(object):

    def __init__(self, model, params):
        self.model = model
        self.params = params  # XParams class
 
    @classmethod
    def read(cls, expdir):
        """read from existing experiment
        """
        o = XDir(expdir) # dir structure
        model = Model.read(o.path("config.json"))
        params = XParams.read(o.path("params.txt"))
        return cls(model, params)

    def write(self, expdir, force=False):
        """Write experiment params and default model to directory
        """
        x = XDir(expdir)
        x.create_expdir(force)
        self.params.write(x.path("params.txt"))
        self.model.setup(x.path("default")) # default model setup


    def run(self, runid=None, background=False, submit=False, dry_run=False, expdir='./', **kwargs):
        """Run a model instance

        Returns
        -------
        Popen instance (or equivalent if submit)
        """
        x = XDir(expdir)
        rundir = x.rundir(runid) # will be created upon run

        logdir = x.logdir()
        if not os.path.exists(logdir): 
            os.makedirs(logdir) # needs to be created before

        logout = kwargs.pop("output", x.logout(runid))
        logerr = kwargs.pop("error", x.logerr(runid))

        if background:
            stdout = open(logout, 'w')
            stderr = open(logerr, 'w')
        else:
            stdout = subprocess.STDOUT
            stderr = subprocess.STDERR

        # environment variables to define
        context = dict(
            runid = runid,
            expdir = expdir,
            rundir = rundir,
            runtag = x.runtag(runid),
        )

        # update model parameters, setup directory
        params = self.params.pset_as_dict(runid)
        model = copy.deepcopy(self.model) 
        model.update(params)

        if dry_run:
            print(mode.command(context))
            return

        model.setup(rundir)

        if submit:
            assert 'array' not in kwargs, "batch command for --array"
            model.submit(context, output=logout, error=logerr, env=makenv(context),
                         jobfile=os.path.join(rundir, 'submit.sh'), **kwargs)

        else:
            env = makenv(context, os.environ.copy())
            p = subprocess.Popen(args, env=env, stdin=stdin, stdout=stdout, stderr=stderr)

        if not background:
            ret = p.wait()

        return p


    def batch(self, array=None, expdir="./", wait=True, submit=True, **kwargs):
        """Run ensemble
        """
        N = self.params.size

        # batch command
        if array is None:
            # all params by default
            array = "{}-{}".format(0, N-1) 

        # write config to expdirectory
        self.setup(force=True)  # things are up to date

        # local testing : do not use slurm
        if not submit:
            indices = parse_slurm_array_indices(array)
            print("Run",len(indices),"out of",N,"simulations in the background")
            print(indices)
            processes = []
            for runid in indices:
                p = self.run(runid=runid, expdir=expdir, background=True)
                processes.append(p)
            if wait:
                for p in processes:
                    p.wait()
            return processes

        # submit job to slurm (the default)
        print("Submit job array batch to SLURM")
        jobfile = os.path.join(expdir, "batch.sh")

        logdir = x.logdir()
        logout = kwargs.pop("output", x.logout("%a"))
        logerr = kwargs.pop("error", x.logerr("%a"))

        # log-files
        if not os.path.exists(logdir): 
            os.makedirs(logdir) # needs to be created before

        # actual command
        pycmd = ["from "+__name__+" import XRun", 
                 "XRun.run(runid=$SLURM_ARRAY_TASK_ID, expdir='{expdir}')"]

        pycmds = "; ".join(pycmd).format( expdir=expdir )

        cmds = '{} -c "{}"'.format(sys.executable, pycmds)
        jobfile = kwargs.pop("jobfile", os.path.join(expdir, 'submit.sh'))

        p = submit_job(cmds, output=logout, error=logerr, jobfile=jobfile **kwargs)

        if wait:
            p.wait()

        return p



##########
#
# Main run
#
from simtools.xparams import ParamsParser


class XParser(object):
    """Helper class to build ArgumentParser with subcommand and keep clean
    """
    def __init__(self, *args, **kwargs):
        self.parser = argparse.ArgumentParser(*args, **kwargs)
        self.subparsers = self.parser.add_subparsers(dest='cmd')

        # define model parser
    def add_model_group(self, root, required=False):
        grp = root.add_argument_group("model")
        grp.add_argument("--config", default="config.json", 
                         help="job configuration file (default=%(default)s)")
        grp.add_argument("-x","--executable", required=required,
                         help="model executable")
        grp.add_argument("--args", default=[], nargs="*",
                         help="model arguments to replace whatever is in the config. Use the `--` separator to append arguments.)")


    def add_config(self, prior):
        """edit configuration file from command-line
        """
        p = self.subparsers.add_parser('config', 
                                       add_help=False, 
                                       parents=[prior],
                                       help=self.add_config.__doc__)


        p.add_argument("--force","-f", action="store_true", 
                       help="do not prompt even if file already exists (edit anyway)")

        self.add_model_group(p, required=True)
        grp = p.add_argument_group("params i/o")

        from simtools.model.params import filetypes
        grp.add_argument("--filetype", choices=filetypes.keys(), 
                                help="model params file type")
        grp.add_argument("--default", 
                        help="default parameters for look-up and checking on parameter names. A file name with same format as specified by --filetype")
        #grp.add_argument("--addon", nargs="*",
        #                help="add ons to import ")

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

        # obs
        grp = p.add_argument_group("constraints")
        p.add_argument("--obs", action="store_true",
                                help="observations to constrain the model")


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


def make_config(args):
    """Make configuration from argument parser
    """

    # model config
    # ------------
    if args.no_args:
        args.params_args = None

    if args.no_write:
        args.params_write = None

    model = {
        "executable" : args.executable,
        "args" : args.args,
        "params_write" : args.params_write,
        "params_args" : args.params_args,
        "filetype" : args.filetype,
        "default" : args.default,
    }

    # check
    modelobj = Model.fromconfig(model)

    # prior params
    # ------------
    if args.prior_file and os.path.exists(args.prior_file):
        prior = json.load(open(args.prior_file))
        assert args.prior_params, "cannot provide both prior_file and prior_params"

    else:
        prior = {
            "params" : [],
        }
        for p in args.prior_params:
            pdef = p.todict()
            prior["params"].append(pdef)

    # update default values, for info
    names = [p.name for p in modelobj.params]
    for p in prior["params"]:
        if p["name"] in names:
            i = names.index(p["name"])
            pval = modelobj.params[i].default 
            phelp = modelobj.params[i].help
            if pval:
                p["default"] = pval
            if help:
                p["help"] = phelp

    # observations
    # ------------
    if args.obs_file and os.path.exists(args.obs_file):
        obs = json.load(open(args.obs_file))
        assert args.obs, "cannot provide both obs_file and obs"

    else:
        obs = {
            "obs" : [],
            #"correlation" : None,
        }
        for p in args.obs:
            pdef = p.todict()
            prior["obs"].append(pdef)

    # Done !! write down to disk !
    config = {
        "model" : model,
        "prior" : prior,
        "obs" : obs,
    }

    return config


def main(argv=None):

    pparser = ParamsParser(description=__doc__, add_help=False)
    
    pp = pparser.add_product()
    ps = pparser.add_sample()
    pr = pparser.add_resample()
    pn = pparser.add_neff()

    xparser = XParser(description=__doc__)

    xparser.add_config(pparser.prior)
    xparser.add_run()
    xparser.add_batch()
    #parser.add_loglik()
    #parser.add_analysis()
    #parser.add_iis()

    args = xparser.parser.parse_args(argv)

    if args.cmd == "config":

        if os.path.exists(args.config):
            if not args.force:
                print("please use '--force' to overwrite existing config")
                import sys
                sys.exit(1)

        config = make_config(args)

        with open(args.config, "w") as f:
            json.dump(config, args.config)

    elif args.cmd == "run":

        xrun = XRun.read(args.expdir)
        xrun.run(runid=args.id, dry_run=args.dry_run, background=args.background)

    elif args.cmd == "batch":

        xrun = XRun.read(args.expdir)
        xrun.batch(array=args.array, background=args.background, 
                      qos=args.qos, job_name=args.job_name, account=args.account, time=args.time, wait=args.wait)


#class XParser(XParams):
#    """Experiment Parser
#    """
#    def __init__(self, *args, **kwargs):
#        self.parser = argparse.ArgumentParser(*args, **kwargs)
#        self.subparsers = self.parser.add_subparsers(dest='cmd')
#        self.define_parents()
#
#
#
#def main(argv=None):

#class Runtime(object):
#    """Interface between model and simtools, to be imported by model at runtime
#    """
#    def __init__(self):
#        """ Communicate with environment variable to derive default parameters
#        """
#        expdir = getenv("expdir", None)
#        runid = getenv("runid", None)
#        self.runid = int(runid) if runid else None
#
#        if expdir is not None:
#            xrun = XRun.read(expdir)
#            self._xrun = xrun
#            self.params = xrun.params.pset_as_dict(self.runid)
#            self.rundir = xrun.rundir(self.runid)
#        else:
#            self._xrun = None
#            self.params = {}
#            self.rundir = None
#
#    def getpar(self, name, *failval):
#        """Retrieve experiment parameters
#        """
#        return self.params.copy().pop(name, *failval)
#
#    def setvar(self, name, value):
#        """Set experiment variables
#        """
#        if self._xrun is None:
#            return  # simtools is not active, do nothing
#        self._xrun.write_state_var(name, value)
