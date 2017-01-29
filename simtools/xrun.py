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
XPARAM = 'params.txt'


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
        #fmt = "{:0>"+str(self.digit)+"}"
        fmt = "{}" # just use integers
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
            #print("error :: directory already exists: "+repr(self.expdir))
            #print("     set  'force' option to bypass this check")
            raise RuntimeError(repr(self.expdir)+" experiment directory already exists")

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
        return XParams.read(self.path(XPARAM)).size


class XRun(object):

    def __init__(self, model, params, autodir=False):
        self.model = model
        self.params = params  # XParams class
        self.autodir = autodir
 
    def setup(self, expdir, force=False):
        """Write experiment params and default model to directory
        """
        x = XDir(expdir)
        x.create_expdir(force)
        self.params.write(x.path(XPARAM))
        self.model.setup(x.path("default")) # default model setup

    def get_rundir(self, runid, expdir):
        x = XDir(expdir)
        if self.autodir:
            params = self.params.pset_as_dict(runid)
            rundir = x.autodir([Param(name, params[name]) for name in self.params.names])
        else:
            rundir = x.rundir(runid)
        return rundir

    def get_model(self, runid):
        """return model
        **context : rundir, used to fill tags in model
        """
        params = self.params.pset_as_dict(runid)
        # update model parameters, setup directory
        model = copy.deepcopy(self.model) 
        model.update(params, context={'runid':runid})
        return model


    def run(self, indices=None, expdir="./", submit=False, include_default=False, **kwargs):
        """Run ensemble
        """
        N = self.params.size
        if indices is None:
            indices = xrange(self.params.size)

        if include_default:
            indices = [None] + np.asarray(indices).tolist()
            bla = ('+ default',)
        else:
            bla = ()

        print("Submit" if submit else "Run",len(indices),"out of",N,"simulations",*bla)
        processes = []
        for runid in indices:
            model = self.get_model(runid)
            rundir = self.get_rundir(runid, expdir)
            if submit:
                p = model.submit(rundir, **kwargs)
            else:
                p = model.run(rundir, background=True)
            processes.append(p)
        return MultiProcess(processes) # has a `wait` command


    def apply(self, func, expdir=None, shp=()):
        """Apply a function on all ensemble members 
        
        * func: callable ( model, rundir ) --> scalar or ndarray
        * expdir : experiment directory
        * shp : shape of the result, by default scalar

        Returns a numpy array with first dimension N (number of models)
        """
        N = self.params.size
        values = np.empty((N,) + shp)
        values.fill(np.nan)

        for i in xrange(N):
            model = self.get_model(i)
            rundir = self.get_rundir(i, expdir)
            try:
                res = func(model, rundir)
            except NotImplementedError:
                raise
            except ValueError:
                continue
            values[i] = res
        return values


    def getvar(self, name, expdir='./'):
        " return one state variable "
        func = lambda model, rundir : model.getvar(name, rundir)
        return self.apply(func, expdir)

    def getstate(self, names, expdir='./'):
        " return many state variable "
        func = lambda model, rundir : [model.getvar(name, rundir) for name in names]
        values = self.apply(func, expdir, shp=(len(names),))
        return XState(values, names)

    def getcost(self, expdir='./'):
        " return cost function "
        func = lambda model, rundir : model.getcost(rundir)
        return self.apply(func, expdir)


class XState(XParams):
    " store model state "
    pass


class MultiProcess(object):
    def __init__(self, processes):
        self.processes = processes

    def apply_many(name, *args, **kwargs):
        return [getattr(p, name)(p, *args, **kwargs) for p in self.processes]

    def wait(self):
        return self.apply_many("wait")
