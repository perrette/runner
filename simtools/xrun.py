"""Experiment run
"""
from __future__ import print_function, absolute_import
import numpy as np
import logging
import json
import copy
import os
import sys
import subprocess
from os.path import join

from simtools.model import Param, Model
from simtools.tools import autofolder, Namespace
from simtools.xparams import XParams

XPARAM = 'params.txt'


# Ensemble Xperiment
# ==================

class XData(XParams):
    " store model state and other data"
    pass


class MultiProcess(object):
    def __init__(self, processes):
        self.processes = processes

    def apply_many(name, *args, **kwargs):
        return [getattr(p, name)(p, *args, **kwargs) for p in self.processes]

    def wait(self):
        return self.apply_many("wait")


class XRun(object):

    def __init__(self, model, params, autodir=False, rundir_template='{}'):
        self.model = model
        self.params = params  # XParams class
        self.autodir = autodir
        self.rundir_template = rundir_template
 
    def setup(self, expdir, force=False):
        """Write experiment params and default model to directory
        """
        if not os.path.exists(expdir):
            logging.info("create directory",expdir)
            os.makedirs(expdir)
        elif not force:
            raise RuntimeError(repr(expdir)+" experiment directory already exists")
        self.params.write(join(expdir, XPARAM))
        try:
            self.model.setup(join(expdir, 'default'))
        except KeyError:
            logging.warn("Failed to setup default model version" +
                  "probably because no default values have been specified" +
                  "and {NAME} syntax was used for command line arguments." +
                  "Nevermind just skip this step.")

    def get_rundir(self, runid, expdir):
        if runid is None:
            return join(expdir, 'default')

        if self.autodir:
            #raise NotImplementedError('autodir')
            params = [Namespace(name=name, value=value) 
                      for name,value in zip(self.params.names, 
                                            self.params.pset_as_array(runid))]
            rundir = join(expdir, autofolder(params))
        else:
            rundir = join(expdir, self.rundir_template.format(runid))
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


    def apply(self, func, expdir=None, indices=None):
        """Apply a function on all member (basically run or submit)
        """
        N = self.params.size
        if indices is None:
            indices = xrange(N)

        results = []
        for i in indices:
            model = self.get_model(i)
            rundir = self.get_rundir(i, expdir)
            ret = func(model, rundir)
            results.append(ret)

        return results


    def run(self, indices=None, expdir="./"):
        """run model in the background
        """
        func = lambda model, rundir : model.run(rundir, background=True)
        return MultiProcess( self.apply(func, expdir, indices))


    def submit(self, indices=None, expdir="./", **kwargs):
        """submit model
        """
        func = lambda model, rundir : model.submit(rundir, **kwargs)
        return MultiProcess( self.apply(func, expdir, indices) )


    def apply_get(self, func, expdir=None, shp=()):
        """Call getvar, getcost etc... on all ensemble members
        
        * func: callable ( model, rundir ) --> scalar or ndarray
        * expdir : experiment directory
        * shp : shape of the result, by default scalar

        Returns a numpy array with first dimension N (number of models)
        RuntimeError are ignored, other are raised
        """
        N = self.params.size
        values = np.empty((N,) + shp)
        values.fill(np.nan)

        for i in xrange(N):
            model = self.get_model(i)
            rundir = self.get_rundir(i, expdir)
            try:
                res = func(model, rundir)
            except RuntimeError:
                continue
            except NotImplementedError:
                raise
            except:
                raise
            values[i] = res
        return values


    def getvar(self, name, expdir='./'):
        " return one state variable "
        func = lambda model, rundir : model.getvar(name, rundir)
        return self.apply_get(func, expdir)

    def getstate(self, names, expdir='./'):
        " return many state variable "
        func = lambda model, rundir : [model.getvar(name, rundir) for name in names]
        values = self.apply_get(func, expdir, shp=(len(names),))
        return XData(values, names)

    def getcost(self, expdir='./'):
        " return cost function "
        func = lambda model, rundir : model.getcost(rundir)
        return self.apply_get(func, expdir)
