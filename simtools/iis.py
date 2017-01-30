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
from simtools.xparams import RESAMPLING_METHOD
from simtools.xrun import XRun, XParams



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
    def __init__(self, model, initdir, constraints, iter=0, epsilon=None, resampling=RESAMPLING_METHOD):
        self.model = model
        self.initdir = initdir
        self.constraints = constraints
        self.iter = iter
        self.epsilon = epsilon
        self.resampling = resampling

    def is_analyzed(self, iter=None):
        return os.path.exists(self.path("loglik.txt", iter))

    def goto_last_iter(self):
        while self.is_analyzed():
            self.iter += 1

    def expdir(self, iter=None):
        iter = self.iter if iter is None else iter
        return self.initdir + ('.'+str(iter)) if iter > 0 else ""

    def path(self, file, iter=None):
        return os.path.join(self.expdir(iter), file)

    def xrun(self, iter=None):
        return XRun(self.model, XParams.read(self.path("params.txt", iter)))

    def resample(self, iter, **kwargs):
        xrun = self.xrun(iter)
        w = np.exp(np.loadtxt(xrun.path("loglik.txt")))

        opt = dict(epsilon=self.epsilon, method=self.resampling)
        opt.update(kwargs)
        xrun.params = xrun.params.resample(weights, **opt)

        xrun.expdir = self.expdir(iter+1)
        return xrun

    def step(self, **kwargs):

        print("******** runiis iter={}".format(self.iter))
        assert not self.is_analyzed(), 'already analyzed'

        if self.iter == 0:
            print("*** first iteration")
            xrun = self.xun()
        else:
            print("*** resample")
            xrun = self.resample(self.iter-1)

        print("*** runbatch")
        xrun.runbatch(wait=True, **kwargs)
        print("*** analysis")
        xrun.analyze(self.constraints).write(xrun.expdir)

        # increment iterations and recursive call
        self.iter += 1

    def runiis(self, maxiter, **kwargs):
        """Iterative Importance Sampling (recursive)
        """
        while self.iter < maxiter:
            self.step(**kwargs)

        print("******** runiis terminated")



