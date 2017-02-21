"""Experiment run
"""
from __future__ import print_function, absolute_import
import logging
import json
import copy
import os
import sys
from multiprocessing import Pool
import six
from os.path import join
import numpy as np

from runner.model import Param, Model
from runner.tools import autofolder, Namespace, nans
from runner.xparams import XParams

XPARAM = 'params.txt'


# Ensemble Xperiment
# ==================

class XData(XParams):
    " store model state and other data"
    pass


class XRun(object):

    def __init__(self, model, params, expdir='./', autodir=False, rundir_template='{}', max_workers=None, chunksize=None):
        self.model = model
        self.params = params  # XParams class
        self.expdir = expdir
        self.autodir = autodir
        self.rundir_template = rundir_template
        self.pool = multiprocessing.Pool(max_workers or params.size)
        self.chunksize = chunksize
 
    def setup(self, force=False):
        """Create directory and write experiment params
        """
        if not os.path.exists(self.expdir):
            logging.info("create directory: "+self.expdir)
            os.makedirs(self.expdir)

        if os.path.exists(join(self.expdir, XPARAM)) and not force:
            raise RuntimeError(repr(self.expdir)+" experiment directory already exists")
        self.params.write(join(self.expdir, XPARAM))
        #try:
        #    self.model.setup(join(expdir, 'default'))
        #except KeyError:
        #    logging.warn("Failed to setup default model version" +
        #          "probably because no default values have been specified" +
        #          "and {NAME} syntax was used for command line arguments." +
        #          "Nevermind just skip this step.")

    def get_rundir(self, runid):
        if runid is None:
            return join(self.expdir, 'default')

        if self.autodir:
            #raise NotImplementedError('autodir')
            params = [Namespace(name=name, value=value) 
                      for name,value in zip(self.params.names, 
                                            self.params.pset_as_array(runid))]
            rundir = join(self.expdir, autofolder(params))
        else:
            rundir = join(self.expdir, self.rundir_template.format(runid))
        return rundir

    def __getitem__(self, runid):
        " return frozen model "
        rundir = self.get_rundir(runid)
        params = self.params.pset_as_dict(runid)
        return self.model(rundir, params)


    def __len__(self):
        return self.params.size


    def __iter__(self):
        #return six.moves.range(self.params.size)
        for i in six.moves.range(self.params.size):
            yield self[i]


    def map_async(self, func, indices=None, callback=None):
        """Wrapper for XRun.pool.map_async
        """
        if indices is None:
            indices = six.moves.range(self.params.size)
        return self.pool.map_async(func, indices, chunksize=self.chunksize, callback=callback)


    def map_model(self, method, indices=None, args=(), **kwargs):
        """call FrozenModel method
        """
        def func(runid):
            try:
                return getattr(self[runid], method)(*args, **kwargs)
            except RuntimeError as error:
                return None
        return self.map_async(func, indices)


    def run(self, indices=None, **kwargs):
        """Run model via multiprocessing.Pool.map_async, and return async result
        """
        return self.map_model("run", indices, **kwargs)


    def postprocess(self):
        r = self.map_model("postprocess")
        return r.get()


    def get_first_valid(self):
        for i, m in enumerate(self):
            if m.load().status == 'success':
                return i
        raise ValueError("no successful run")


    def get_output_names(self):
        return self[self.get_first_valid()].load().output.keys()


    def get_output(self, names=None):
        if names is None:
            names = self.get_output_names()

        def func(runid):
            m = self[runid].load()
            if m.status == "success": 
                return [m.output[nm] for nm in names]
            else:
                return [np.nan for nm in names]

        values = np.asarray(self.map_async(func).get())
        return XData(values, names)


    def get_params(self, names=None):
        " for checking only "
        if names is None:
            return self[self.get_first_valid()].load().params.keys()

        def func(runid):
            m = self[runid].load()
            return [m.params[nm] for nm in names]

        values = np.asarray(self.map_async(func).get())
        return XData(values, names)


    def get_loglik(self):

        def func(runid):
            m = self[runid].load()
            if m.status == "success": 
                return m.likelihood.logpdf()
            else:
                return [np.nan for nm in m.likelihood]

        names = self.model.likelihood.names
        values = np.asarray(self.map_async(func).get())
        return XData(values, names)


    def get_weight(self):
        def func(runid):
            m = self[runid].load()
            if m.status == "success": 
                return np.exp(m.likelihood.logpdf().sum())
            else:
                return 0
        return np.asarray(self.map_async(func).get())


    def get_valids(self, alpha):

        def func(runid):
            m = self[runid].load()
            if m.status == "success": 
                return m.likelihood.isvalid(alpha)
            else:
                return [False for nm in m.likelihood]

        names = self.model.likelihood.names
        values = np.asarray(self.map_async(func).get())
        return XData(values, names)


    def get_valid(self, alpha):
        def func(runid):
            m = self[runid].load()
            if m.status == "success": 
                return m.likelihood.isvalid(alpha).all()
            else:
                return False
        return np.asarray(self.map_async(func).get())
