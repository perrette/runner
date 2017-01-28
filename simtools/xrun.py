"""Experiment run
"""
from __future__ import print_function, absolute_import
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

## prefix for environ variables

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

    def path(self, base, *args):
        return os.path.join(self.expdir, base, *args)

    def autodir(self, params):
        """automatic directory name based on parameter names
        """
        raise NotImplementedError('autodir')

    def runtag(self, runid=None):
        "provide a tag with same length accross the ensemble"
        digit = self.digit or len(str(self.size()-1))
        fmt = "{:0>"+str(self.digit)+"}"
        return fmt.format(runid) if runid is not None else "default"

    def rundir(self, runid=None):
        if runid is not None:
            runtag = self.runtag(runid)
            rundirs = [runtag]
            #rundirs = _create_dirtree(runtag)
            return self.path(*rundirs)
        else:
            return self.path("default")

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
 
    def setup(self, expdir, force=False):
        """Write experiment params and default model to directory
        """
        x = XDir(expdir)
        x.create_expdir(force)
        self.params.write(x.path("params.txt"))
        self.model.setup(x.path("default")) # default model setup


    def run(self, runid=None, background=False, submit=False, dry_run=False, 
            expdir='./', rundir=None, output=None, error=None, **kwargs):
        """Run one model instance

        runid : which model instance
        rundir : None (--> runid) or a string containing any tag or '--auto'

        Returns
        -------
        Popen instance (or equivalent if submit)
        """
        x = XDir(expdir)

        params = self.params.pset_as_dict(runid)

        # determine run directory and tag (basically the basename of rundir)
        if rundir is None:
            rundir = x.rundir(runid) # will be created upon run
            runtag = x.runtag(runid)
        elif rundir == '--auto':
            rundir = x.autodir([Param(name, params[name]) for name in self.params.names])
            runtag = os.path.basename(rundir)
        else:
            runtag = os.path.basename(rundir)
            #rundir = os.path.join(expdir, rundir)

        # determine log file names
        logdir = os.path.join(expdir, 'logs')
        if not os.path.exists(logdir): 
            os.makedirs(logdir) # needs to be created before
        output = output or os.path.join(logdir, '{runtag}.out').format(runtag=runtag)
        error = error or os.path.join(logdir, '{runtag}.err').format(runtag=runtag)

        # open files for Popen (not used if `submit`)
        if background:
            stdout = open(output, 'w')
            stderr = open(error, 'w')
        else:
            stdout = None
            stderr = None

        # environment variables to define and tag fillers
        context = dict(
            runid = runid,
            runtag = runtag,
            expdir = expdir,
        )

        # update model parameters, setup directory
        model = copy.deepcopy(self.model) 
        model.update(params)

        if dry_run:
            print(model.command(rundir, context))
            return

        model.setup(rundir)

        if submit:
            assert 'array' not in kwargs, "batch command for --array"
            p = model.submit(rundir, context, output=output, error=error,
                         jobfile=os.path.join(rundir, 'submit.sh'), **kwargs)

        else:
            p = model.run(rundir, context=context, stdout=stdout, stderr=stderr)

        if not background:
            ret = p.wait()

        return p


    def batch(self, indices=None, expdir="./", submit=True, autodir=False, include_default=False, **kwargs):
        """Run ensemble
        """
        N = self.params.size
        # write config to expdirectory
        # self.setup(force=True)  # things are up to date

        if indices is None:
            indices = np.arange(self.params.size).tolist()

        if include_default:
            indices = indices + [None]
            bla = ('+ default',)
        else:
            bla = ()

        print("Submit" if submit else "Run",len(indices),"out of",N,"simulations",*bla)
        processes = []

        if autodir:
            rundir='--auto'
        else:
            rundir=None

        for runid in indices:
            p = self.run(runid=runid, expdir=expdir, submit=submit, background=True, rundir=rundir, **kwargs)
            processes.append(p)
        return MultiProcess(processes) # has a `wait` command


class MultiProcess(object):
    def __init__(self, processes):
        self.processes = processes

    def apply_many(name, *args, **kwargs):
        return [getattr(p, name)(p, *args, **kwargs) for p in self.processes]

    def wait(self):
        return self.apply_many("wait")
