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


class _AbortableWorker(object):
    """ to handle TimeOut individually and freeup ressources
    http://stackoverflow.com/a/29495039/2192272
    """
    def __init__(self, func, timeout=None):
        self.func = func
        self.timeout = timeout

    def __call__(self, *args, **kwargs):
        p = ThreadPool(1)
        res = p.apply_async(self.func, args=args, kwds=kwargs)
        try:
            out = res.get(self.timeout)  # Wait timeout seconds for func to complete.
            return out
        except multiprocessing.TimeoutError:
            p.terminate()
            raise


class _PickableMethod(_AbortableWorker):
    """ make a class method pickable (because defined at module-level) 
    for use in multiprocessing
    """
    def __init__(self, obj, method, timeout=None):
        self.obj = obj
        self.method = method
        self.timeout = timeout

    @property
    def func(self):
        return getattr(self.obj, self.method)


class XRun(object):

    def __init__(self, model, params, expdir='./', autodir=False, rundir_template='{}', max_workers=None, timeout=31536000):
        self.model = model
        self.params = params  # XParams class
        self.expdir = expdir
        self.autodir = autodir
        self.rundir_template = rundir_template
        self.max_workers = max_workers
        self.timeout = timeout
 
    def setup(self, force=False):
        """Create directory and write experiment params
        """
        if not os.path.exists(self.expdir):
            logging.info("create directory: "+self.expdir)
            os.makedirs(self.expdir)

        pfile = join(self.expdir, XPARAM)
        if os.path.exists(pfile) and not force:
            raise RuntimeError(repr(pfile)+" param file already exists")
        self.params.write(join(self.expdir, XPARAM))

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


    def run(self, indices=None, callback=None, **kwargs):
        """Wrapper for multiprocessing.Pool.map
        """
        if indices is None:
            N = len(self)
            indices = six.moves.range(N)
        else:
            N = len(indices)
        pool = multiprocessing.Pool(self.max_workers or N, init_worker)

        # submit
        ares = []
        for model in self:
            run_model = _PickableMethod(model, 'run', timeout=self.timeout)
            r = pool.apply_async(run_model, kwds=kwargs, callback=callback)
            ares.append(r)

        # get result and handle errors individually
        res = []
        sucesses = 0
        try:
            for i, r in zip(indices, ares):
                try:
                    res.append( r.get() )
                    sucesses += 1
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

        if sucesses > 0:
            logging.info("{} out of {} runs completed successfully".format(sucesses, N))
        else:
            logging.error("all runs failed")

        return res


    def postprocess(self):
        return [m.postprocess() if m.load().status == "success" else None 
                for m in self]


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
