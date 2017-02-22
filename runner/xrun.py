"""Experiment run
"""
from __future__ import print_function, absolute_import
import logging
import signal
import time
import json
import copy
import os
import sys
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
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


def _model_output_as_array(m, names=None):
    if m.status == "success": 
        if names is None:
            names = m.output
        res = [m.output[nm] if not np.ndim(m.output[nm]) else np.mean(m.output[nm]) for nm in names]
    else:
        res = np.nan
    return res


def init_worker():
    # to handle KeyboardInterrupt manually
    # http://stackoverflow.com/a/6191991/2192272
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def abortable_worker(func, *args, **kwargs):
    # to handle TimeOut individually and freeup ressources
	# http://stackoverflow.com/a/29495039/2192272

    timeout = kwargs.pop('timeout', None)
    p = ThreadPool(1)
    res = p.apply_async(func, args=args, kwds=kwargs)
    try:
        out = res.get(timeout)  # Wait timeout seconds for func to complete.
        return out
    except multiprocessing.TimeoutError:
        p.terminate()
        raise


class XRun(object):

    def __init__(self, model, params, expdir='./', autodir=False, rundir_template='{}', max_workers=None, chunksize=None, timeout=31536000):
        self.model = model
        self.params = params  # XParams class
        self.expdir = expdir
        self.autodir = autodir
        self.rundir_template = rundir_template
        self.max_workers = max_workers
        self.chunksize = chunksize or 1
        self.timeout = timeout
 
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


    def map(self, func, indices=None, kwds={}, callback=None):
        """Wrapper for multiprocessing.Pool.map
        """
        if indices is None:
            N = len(self)
            indices = six.moves.range(N)
        else:
            N = len(indices)
        pool = multiprocessing.Pool(self.max_workers or max(1,N//self.chunksize), 
                                    init_worker)

        # submit
        ares = []
        for i in indices:
            ares.append(pool.apply_async(func, (i,), kwds, callback))

        # get result and handle errors individually
        res = []
        try:
            for i, r in zip(indices, ares):
                try:
                    res.append( r.get() )
                except multiprocessing.TimeoutError:
                    logging.warn('TIMEOUT: '+str(i))
                    res.append( None )
                except KeyboardInterrupt:
                    raise
                except Exception as error:
                    logging.warn(str(i)+' :: '+str(error))
                    res.append( None )

        finally:
            # kill any remaining task
            pool.terminate()
            pool.close()
        return res

    def map_model(self, method, indices=None, args=(), check=True, **kwargs):
        """call FrozenModel method
        """
        func = ModelFunc(self, method, args=args, kwargs=kwargs, timeout=self.timeout)
        res = self.map(func, indices)
        if check:
            anygood = False
            for m in res:
                if m is not None:
                    anygood = True
                    break
            if not anygood:
                raise RuntimeError('all model run failed')
        return res


    def run(self, indices=None, **kwargs):
        """Run model via multiprocessing.Pool.map
        """
        return self.map_model("run", indices, **kwargs)


    def postprocess(self):
        return self.map_model("postprocess")


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
        values = nans((len(self), len(names)))
        for i, m in enumerate(self):
            m.load()
            values[i] = _model_output_as_array(m, names)

        return XData(values, names)


    def get_params(self, names=None):
        " for checking only "
        if names is None:
            return self[self.get_first_valid()].load().params.keys()
        values = np.empty((len(self), len(names)))
        for i, m in enumerate(self):
            m.load()
            values[i] = [m.params[nm] for nm in names]
        return XData(values, names)


    def get_logliks(self):
        names = self.model.likelihood.names
        values = nans((len(self), len(names)))
        for i, m in enumerate(self):
            m.load()
            if m.status == "success": 
                values[i] = m.likelihood.logpdf()
        return XData(values, names)


    def get_weight(self):
        logliks = self.get_logliks().values
        return np.where(np.isnan(logliks), 0, np.exp(logliks.sum(axis=1)))


    def get_valids(self, alpha, names=None):
        if names is None:
            names = self.model.likelihood.names

        values = np.zeros((len(self), len(names)), dtype=bool)
        for i, m in enumerate(self):
            m.load()
            if m.status != "success": 
                continue
            if alpha is None:
                values[i] = True
            else:
                values[i] = [m.likelihood[name].isvalid(alpha) for name in names]
        return XData(values, names)


    def get_valid(self, alpha=None, names=None):
        return self.get_valids(alpha, names).values.all(axis=1)


class ModelFunc(object):
    """pickable function for multiprocessing.Pool
    """
    def __init__(self, xrun, method, args=(), kwargs={}, timeout=None, load=False):
        self.xrun = xrun
        self.load = load
        self.method = method
        self.args = args
        self.kwargs = kwargs
        self.timeout = timeout

    def __call__(self, runid):
        model = self.xrun[runid]
        if self.load:
            model.load()
        func = getattr(model, self.method)
        #return func(*self.args, **self.kwargs)
        return abortable_worker(func, *self.args, timeout=self.timeout, **self.kwargs)
