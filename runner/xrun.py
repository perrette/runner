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
import six
from os.path import join
import numpy as np

from runner.tools.tree import autofolder
from runner.tools.frame import str_dataframe
from runner.model import Param, Model
from runner.xparams import XParams

XPARAM = 'params.txt'

def nans(N):
    a = np.empty(N)
    a.fill(np.nan)
    return a



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
        from multiprocessing.dummy import Pool as ThreadPool
        p = ThreadPool(1)
        res = p.apply_async(self.func, args=args, kwds=kwargs)
        try:
            out = res.get(self.timeout)  # Wait timeout seconds for func to complete.
            return out
        except multiprocessing.TimeoutError:
            p.terminate()
            raise multiprocessing.TimeoutError(str(self.timeout))


class _PickableMethod(object):
    """ make a class method pickable (because defined at module-level) 
    for use in multiprocessing
    """
    def __init__(self, obj, method):
        self.obj = obj
        self.method = method

    def __call__(self, *args, **kwargs):
        return getattr(self.obj, self.method)(*args, **kwargs)


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

        # model params
        pfile = self.path(XPARAM)
        if os.path.exists(pfile) and not force:
            raise RuntimeError(repr(pfile)+" param file already exists")
        self.params.write(self.path(XPARAM))

        # model interface
        self.model.write(self.expdir, force)

        # directory structure
        with open(self.path('xrun.json'), 'w') as f:
            json.dump({k:v for k,v in vars(self).items() 
                       if k not in ['model', 'params', 'expdir']}, f)

    @classmethod
    def load(cls, expdir='./'): 
        model = Model.read(expdir)
        params = XParams.read(join(expdir, XPARAM))
        kwds = json.load(open(join(expdir, 'xrun.json')))
        return cls(model, params, expdir, **kwds)

    def path(self, *args):
        return os.path.join(self.expdir, *args)

    def get_rundir(self, runid):
        if runid is None:
            return join(self.expdir, 'default')

        if self.autodir:
            #raise NotImplementedError('autodir')
            params = [(name,value)
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


    def _run(self, i, **kwargs):
        return self[i].run(**kwargs)

    def run(self, indices=None, callback=None, **kwargs):
        """Wrapper for multiprocessing.Pool.map
        """
        if indices is None:
            N = len(self)
            indices = six.moves.range(N)
        else:
            N = len(indices)

        # workers pool
        pool = multiprocessing.Pool(self.max_workers or N, init_worker)

        # prepare method
        run_model = _PickableMethod(self, '_run')
        run_model = _AbortableWorker(run_model, timeout=self.timeout)

        ares = [pool.apply_async(run_model, (i,), kwds=kwargs, callback=callback) for i in indices]

        res = []
        successes = 0
        for i,r in enumerate(ares):
            try:
                res.append(r.get(1e9))
                successes += 1
            except Exception as error:
                logging.warn("run {} failed:{}:{}".format(i, type(error).__name__, str(error)))
                res.append(None)

        if successes == N:
            logging.info("all runs finished successfully")

        elif successes > 0:
            logging.warn("{} out of {} runs completed successfully".format(successes, N))
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


    def _get_params(self, names=None):
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


    def get_likelihood(self):
        values = np.zeros(len(self), dtype=float)
        for i, m in enumerate(self):
            m.load()
            if m.status == "success": 
                values[i] = np.exp(m.likelihood.logpdf().sum())
        return values


    def get_prior(self):
        if not self.model.prior:
            return 1
        values = np.ones(len(self), dtype=float)
        for i, m in enumerate(self):
            values[i] = np.exp(m.prior.logpdf().sum())
        return values


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


    def analyze(self, names=None, anadir=None):
        """Perform analysis of the ensemble (write to disk)
        """
        if anadir is None:
            anadir = self.expdir

        # Check number of valid runs
        print("Experiment directory: "+self.expdir)
        print("Total number of runs: {}".format(len(self)))
        print("Number of successful runs: {}".format(self.get_valid().sum()))


        # Check outputs
        # =============
        names = names or []
        names = names + [x.name for x in self.model.likelihood 
                                 if x.name not in names]
        if not names:
            names = self.get_output_names()
            logging.info("Detected output variables: "+", ".join(names))


        # Write output variables
        # ======================
        if names:
            xoutput = self.get_output(names)
        else:
            xoutput = None

        if xoutput is not None:
            outputfile = os.path.join(anadir, "output.txt")
            logging.info("Write output variables to "+outputfile)
            xoutput.write(outputfile)

        # Derive likelihoods
        # ==================
        xlogliks = self.get_logliks()
        file = os.path.join(anadir, 'logliks.txt')
        logging.info('write logliks to '+ file)
        xlogliks.write(file)

        # Sum-up and apply custom distribution
        # ====================================
        logliksum = xlogliks.values.sum(axis=1)
        file = os.path.join(anadir, "loglik.txt")
        logging.info('write loglik (total) to '+ file)
        np.savetxt(file, logliksum)

        # Add statistics
        # ==============
        valid = np.isfinite(logliksum)
        ii = [xoutput.names.index(c.name) for c in self.model.likelihood]
        output = xoutput.values[:, ii] # sort !
        pct = lambda p: np.percentile(output[valid], p, axis=0)

        names = [c.name for c in self.model.likelihood]

        #TODO: include parameters in the stats
        #for c in self.model.prior:
        #    if c.name not in self.params.names:
        #        raise ValueError('prior name not in params: '+c.name)

        res = [
            ("obs", [c.dist.mean() for c in self.model.likelihood]),
            ("best", output[np.argmax(logliksum)]),
            ("mean", output[valid].mean(axis=0)),
            ("std", output[valid].std(axis=0)),
            ("min", output[valid].min(axis=0)),
            ("p05", pct(5)),
            ("med", pct(50)),
            ("p95", pct(95)),
            ("max", output[valid].max(axis=0)),
            ("valid_99%", self.get_valids(0.99).values.sum(axis=0)),
            ("valid_67%", self.get_valids(0.67).values.sum(axis=0)),
        ]

        index = [nm for nm,arr in res if arr is not None]
        values = [arr for nm,arr in res if arr is not None]

        stats = str_dataframe(names, values, include_index=True, index=index)

        with open(os.path.join(anadir, 'stats.txt'), 'w') as f:
            f.write(stats)

        #import pandas as pd
        #df = pd.DataFrame(np.array(values), columns=names, index=index)

            #f.write(str(df))
