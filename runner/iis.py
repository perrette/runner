"""Iterative Importance sampling as strategy
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

#from glaciermodel import GlacierModel
from runner.xparams import RESAMPLING_METHOD
from runner.xrun import XRun, XParams



#DIGIT = 4  # number of digits for output folders
#
#    def top_rundirs(self, indices):
#        """top rundir directories for linking
#        """
#        tops = ["default"]
#        for i in indices:
#            top = self.rundir(i).split(os.path.sep)[0]
#            if top not in tops:
#                tops.append(top)
#        return tops
#
#    def link_results(self, orig):
#        """Link results from a previous expdir
#        """
#        assert orig != self.expdir, 'same directories !'
#        print("...link simulations results from",orig)
#        x = XDir(orig)
#        topdirs = x.top_rundirs(xrange(self.size()))
#        for top in topdirs:
#            os.system("cd "+self.expdir+" && ln -s "+os.path.abspath(top))
#    def path(self, base, *args):
#        return os.path.join(self.expdir, base, *args)


class IISExp(object):
    """Handle IIS experiment
    """
    def __init__(self, model, initdir, iter=0, epsilon=None, seed=None, resampling=RESAMPLING_METHOD, size=None):
        self.model = model
        self.initdir = initdir
        self.iter = iter
        self.epsilon = epsilon
        self.resampling = resampling
        self.size = size
        self.seed = seed

    def expdir(self, iter=None):
        iter = self.iter if iter is None else iter
        return self.initdir + ('.'+str(iter)) if iter > 0 else ""

    def path(self, file, iter=None):
        return os.path.join(self.expdir(iter), file)

    def is_analyzed(self, iter=None):
        return os.path.exists(self.path("weights.txt", iter))

    def goto_last_iter(self):
        while self.is_analyzed():
            self.iter += 1

    def xrun(self, iter=None):
        return XRun(self.model, XParams.read(self.path("params.txt", iter)))

    def resample(self, iter, **kwargs):
        xrun = self.xrun(iter)
        w = np.loadtxt(self.path("weights.txt", iter))

        opt = dict(epsilon=self.epsilon, method=self.resampling, size=self.size, seed=self.seed)
        opt.update(kwargs)
        xrun.params = xrun.params.resample(w, **opt)
        xrun.expdir = self.expdir(iter+1)
        xrun.write(self.path("params.txt"))
        return xrun


    def sample(self, size=None, **sampling):
        print("******** iis sampling from prior")
        pfile = self.path("params.txt", 0)
        assert not os.path.exists(pfile), 'already initialized'
        assert size or self.size, 'size required'
        xparam = self.model.prior.sample(size or self.size, seed=self.seed, **sampling)
        xparam.write(pfile)


    def step(self, **kwargs):

        print("******** runiis iter={}".format(self.iter))
        assert not self.is_analyzed(), 'already analyzed'

        if self.iter == 0:
            print("*** first iteration")
            xrun = self.xrun()
        else:
            print("*** resample")
            xrun = self.resample(self.iter-1)

        print("*** run")
        xrun.run(**kwargs)
        print("*** analysis")
        posterior = xrun.get_prior()*xrun.get_likelihood()
        np.savetxt( self.path("weights.txt"), posterior )
        #xrun.analyze()

        # increment iterations and recursive call
        self.iter += 1
        if self.seed is not None:
            self.seed += 1


    def run(self, maxiter, **kwargs):
        while self.iter < maxiter:
            self.step(**kwargs)
        print("******** iis run terminated")


    def start(self, niter, size=None, **kwargs):
        assert self.iter == 0
        self.sample(size)
        self.run(niter, **kwargs)


    def restart(self, niter, size=None, **kwargs):
         self.goto_last_iter()
         if size:
             self.size = size
         self.run(self.iter + niter, **kwargs)
