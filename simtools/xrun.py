"""Experiment run
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

from simtools.modelrun import run_command, parse_slurm_array_indices
from simtools.params import XParams

# prefix for environ variables
ENVPREFIX = "SIMTOOLS_"

ENVIRON_MAP = {
    "runid" : ENVPREFIX+"RUNID",
    "expdir" : ENVPREFIX+"EXPDIR",
}

def getenv(name, *failobj):
    " for runner to set environment variables" 
    return os.environ.get(ENVIRON_MAP[name], *failobj)



#class ParamSet(object):
#    """Represent a parameter set for a model
#    """
#    def __init__(self, params=None):
#        """params : dict of params
#        """
#        if isinstance(params, ParamSet):
#            params = params.params
#        self.params = dict(params)
#
#    @classmethod
#    def parfile(cls, rundir):
#        return os.path.join(rundir, "params.json")
#
#    @classmethod
#    def read(cls, rundir):
#        """Read single from rundir
#        """
#        return cls(json.load(open(cls.parfile(rundir))))
#
#    def write(self, rundir):
#        """write params to rundir
#        """
#        with open(self.parfile(rundir), "w") as f:
#            json.dump(self.params, f)
#
#    def copy(self):
#        return ParamSet(self.params.copy())

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
    def __init__(self, expdir, digit=4):
        self.expdir = expdir
        self.digit = digit

    def path(self, *file):
        return os.path.join(self.expdir, *file)

    def runtag(self, runid=None):
        fmt = "{:0>"+str(self.digit)+"}"
        return fmt.format(runid) if runid is not None else "default"

    def rundir(self, runid=None):
        if runid is not None:
            runtag = self.runtag(runid)
            rundirs = _create_dirtree(runtag)
            return self.path(*rundirs)
        else:
            return self.path("default")

    def statefile(self, runid=None):
        """state variable name
        """
        runtag = self.runtag(runid)
        return self.path(self.expdir, "state.{}".format(runtag))

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


class XRun(XDir):

    def __init__(self, model, params, expdir):
        self.model = model
        self.params = params  # XParams class
        self.expdir = expdir
 
    @classmethod
    def read(cls, expdir):
        """read from existing experiment
        """
        o = XDir(expdir) # dir structure
        params = XParams.read(o.path("params.txt"))
        # also read default parameters
        defaultdir = o.rundir(None)
        try:
            default = json.load(open(os.path.join(defaultdir, "params.json")))
            params.default = [default[nm] for nm in params.names]
        except:
            pass
        return cls(params, expdir)

    def setup(self, newdir=None, force=False):
        """Setup experiment directory
        """
        newdir = newdir or self.expdir
        x = XDir(newdir)
        x.create_expdir(force)
        self.params.write(x.path("params.txt"))

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


    # state variables I/O
    def write_state_var(self, name, value, runid=None):
        """Write state variable on disk in a format understood by XRun
        """
        statefile = self.statefile(runid)+'.json' # state file in json format
        with open(statefile, "w") as f:
            json.dump(value, f)

    def read_state_var(self, name, runid=None):
        statefile = self.statefile(runid)+'.json' # state file in json format
        with open(statefile) as f:
            return json.load(f)


    # analyze ensemble
    # ----------------
    def get(self, name, runid=None):
        """Get variable 
        """
        return self.read_state_var(name, runid)


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
        from simtools.analysis import Results

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


class Runtime(object):
    """Interface between model and simtools, to be imported by model at runtime
    """
    def __init__(self):
        """ Communicate with environment variable to derive default parameters
        """
        expdir = XRun.getenv("expdir", None)
        runid = XRun.getenv("runid", None)
        self.runid = int(runid) if runid else None

        if expdir is not None:
            xrun = XRun.read(expdir)
            self._xrun = xrun
            self.params = xrun.params.pset_as_dict(self.runid)
            self.rundir = xrun.rundir(self.runid)
        else:
            self._xrun = None
            self.params = {}
            self.rundir = None

    def getpar(self, name, *failval):
        """Retrieve experiment parameters
        """
        return self.params.copy().pop(name, *failval)

    def setvar(self, name, value):
        """Set experiment variables
        """
        if self._xrun is None:
            return  # simtools is not active, do nothing
        self._xrun.write_state_var(name, value)
