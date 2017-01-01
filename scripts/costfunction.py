#!/usr/bin/env python2.7
"""Compute cost-function for a model output
"""
from __future__ import print_function, division
import argparse
from argparse import RawDescriptionHelpFormatter
import os, sys, datetime, warnings

#from genparams import str_pmatrix

from itertools import product
from collections import OrderedDict as odict

import dimarray as da  # to write to netCDF
import netCDF4 as nc
import numpy as np
import scipy.stats
from scipy.stats import norm, lognorm, uniform, multivariate_normal


def str_pmatrix(pnames, pmatrix, max_rows=10, include_index=True, index=None):
    """Pretty-print parameters matrix like in pandas, but using only basic python functions
    """
    # determine columns width
    col_width_default = 6
    col_fmt = []
    col_width = []
    for p in pnames:
        w = max(col_width_default, len(p))
        col_width.append( w )
        col_fmt.append( "{:>"+str(w)+"}" )

    # also add index !
    if include_index:
        idx_w = len(str(len(pmatrix)-1)) # width of last line index
        idx_fmt = "{:<"+str(idx_w)+"}" # aligned left
        col_fmt.insert(0, idx_fmt)
        pnames = [""]+list(pnames)
        col_width = [idx_w] + col_width

    line_fmt = " ".join(col_fmt)

    header = line_fmt.format(*pnames)

    # format all lines
    lines = []
    for i, pset in enumerate(pmatrix):
        if include_index:
            ix = i if index is None else index[i]
            pset = [ix] + list(pset)
        lines.append(line_fmt.format(*pset))

    n = len(lines)
    # full print
    if n <= max_rows:
        return "\n".join([header]+lines)

    # partial print
    else:
        sep = line_fmt.format(*['.'*min(3,w) for w in col_width])  # separator '...'
        return "\n".join([header]+lines[:max_rows//2]+[sep]+lines[-max_rows//2:])


def parse_scipydist(string):
    """parse a scipy distribution:  NAME?P1,P2,...

    e.g. norm?0,1 or uniform?-10,10 or lognorm?1,0,1
    """
    dtype, spec = string.split('?')
    params = [float(p) for p in spec.split(',')]
    return getattr(scipy.stats, dtype)(*params)


def parse_indices(spec, literal_indices=['gl','c']):

    idx = spec.split(',')
    n = len(idx)

    idx2 = []
    literal_index = False
    for s in idx:
        if ':' in s:
            # slice
            start, stop, step = s.split(':')
            idx2.extend(range(int(start),int(stop),int(step)))
        else:
            # single index
            if s in literal_indices:
                literal_index = True
            else:
                s = int(s)
            idx2.append(s)

    # convert indices to numpy array if no G.L. or calving front spec
    if len(idx2) == 1:
        idx2 = idx2[0]

    elif not literal_index:
        idx2 = np.array(idx2)

    return idx2



class Constraint(object):
    """Generic constraint class, assumes variable in a netCDF file.
    """
    def __init__(self, name, dist, units="", desc=""):


        self.name = name
        self.units = units
        self.desc = desc
        self.dist = dist

    def read(self, out_dir):
        return read_model(os.path.join(out_dir, 'restart.nc'), self.name)

    def logpdf(self, state):
        return self.dist.logpdf(state)

    def mean(self):
        return self.dist.mean()


class Uniform(Constraint):
    def __init__(self, name, low, hi, units="", desc=""):
        Constraint.__init__(self, name, uniform(low, hi), units, desc) 


class Normal(Constraint):
    def __init__(self, name, mean, sd, units="", desc=""):
        Constraint.__init__(self, name, norm(mean, sd), units, desc) 


class LogNormal(Constraint):
    """scipy.stats.lognorm """
    def __init__(self, name, s, loc, scale, units="", desc=""):
        Constraint.__init__(self, name, lognorm(s, loc, scale), units, desc) 


class RMS(Constraint):
    """Root Mean Square : treat as a single observation
    """
    def __init__(self, name, mean, sd, valid=None, units="", desc=""):
        self.name = name 
        self.units = units
        self.desc = desc
        self.mean = mean
        self.var = sd**2
        self.valid = valid

    def logpdf(self, state):
        # overwrite standard method to handle difference in grid
        #n = min(state.size, self._mean.size)
        n = self.mean.size
        mask = np.ones(n, dtype=bool)
        if self.valid:
            for cond in self.valid.split(','):
                if cond == 'pos':
                    mask = mask & (state > 0) & (self.mean > 0)
                elif cond == 'neg':
                    mask = mask & (state < 0) & (self.mean < 0)
                elif cond == '~nan':
                    mask = mask & (~np.isnan(state)) & (~np.isnan(self.mean))
                else:
                    raise ValueError('unknown value for valid='+str(cond))
        misfit = state[mask] - self.mean[mask]
        var = self.var[mask]
        return -0.5 * np.mean( misfit ** 2 / var ) 


    @classmethod
    def parse(cls, name, type='rms', mean=None, sd=None, mean_var=None, sd_var=None, file=None, valid=None, desc="", units=""):
        """Valid arguments for RMS constraint:

        type=rms
        mean=v1[,v2,...] (comma-separated)
        sd=(2,3,4|10|20%) : standard deviation as comma-separated or constant or as percentage of the mean
        mean_var=V1 variable in netCDF file, requires file=
        sd_var=V2 variable in netCDF file, requires file=
        file=FNAME : file name for mean_var and sd_var
        valid=[pos],[~nan] : conditions for data validity, over which RMS is computed (can be cumulated)
            pos : only positive values
            ~nan : non-nan values
        """
        assert type == 'rms'

        # compare to actual data: mean as variable name in a file
        if mean:
            mean_ = np.array([float(v) for v in mean.split(',')])

        elif file:
            if not mean_var:  # by default, just use the same name
                mean_var = name
            mean_ = read_model(file, mean_var)
        else:
            raise ValueError('need to provide mean deviation as mean= or mean_var= and file=')

        # standard dev as fixed error or a relative error 
        n = mean_.size
        if sd:
            if sd.endswith('%'):
                sd_ = mean_*float(sd[:-1])/100
            else:
                sd_ = np.array([float(m) for m in sd.split(',')])

        elif sd_var:
            assert file
            sd_ = read_model(file, sd_var)

        else:
            raise ValueError('need to provide standard deviation as sd= or sd_var= and file=')

        return cls(name, mean_, sd_, valid=valid, desc=desc, units=units)


class MultiVariateNormal(Constraint):
    """Account for covariances
    """
    def __init__(self, name, mean, cov, units="", desc=""):
        dist = multivariate_normal(mean, cov=cov)
        Constraint.__init__(self, name, dist, units, desc) 



def parse_constraint(*args):
    " name=... type=... units=... desc=... mean=... sd=... mean_var=... sd_var... file=... valid=... etc..."

    kwargs = {}
    for i, arg in enumerate(args):
        nm, val = arg.split('=')
        kwargs[nm] = val

    name = kwargs.pop('name')
    dtype = kwargs.pop('type')

    if '?' in dtype:
        dtype, spec = dtype.split('?')
    else:
        spec = ''
    
    if dtype == 'norm':
        loc, scale = spec.split(',')
        constraint = Normal(name, float(loc), float(scale), **kwargs)

    elif dtype == 'uniform':
        loc, scale = spec.split(',')
        constraint = Uniform(name, float(loc), float(scale), **kwargs)

    elif dtype == 'lognorm':
        s, loc, scale = spec.split(',')
        constraint = LogNormal(name, float(s), float(loc), float(scale), **kwargs)

    elif dtype == 'rms':
        assert not spec, 'invalid specificiation for rms type, use keywords mean, sd, mean_var, sd_var'
        try:
            constraint = RMS.parse(name, dtype, **kwargs)
        except:
            print(RMS.parse.__doc__)
            raise

    else:
        raise ValueError("Unknown constraint type: "+dtype)

    return constraint




def get_ensemble_size(outdir):
    import glob
    direcs = sorted(glob.glob(os.path.join(outdir,'0*/')))
    lastid = os.path.basename(direcs[-1].rstrip('/'))
    return int(lastid)+1  # assum start at 0 (in case some intermediary direcs are missing)


def _get_gl(H, zb, rho_sw=1028, rho_i = 917):
    " get index of last grounded cell "
    Hf = -zb*(rho_sw/rho_i) # flotation height
    return np.where(H > Hf)[0][-1]


def _get_terminus(ds):
    """Get indices for calving front and grounding line, given a netCDF4.Dataset of restart file
    """
    if 'c' in ds['x'].ncattrs():
        c = ds['x'].c - 1 # calving front index in python notation        Hc = ds['H'][c]
    else:
        warnings.warn('calving front index not provided, derive it from H')
        H = ds['H'][:]
        c = np.where(H > 5)[0][-1]

    if 'gl' in ds['x'].ncattrs():
        gl = ds['x'].gl - 1 # calving front index in python notation        Hc = ds['H'][c]
    else:
        warnings.warn('grounding line index not provided, derive from H and zb')
        gl = _get_gl(ds['H'][:c], ds['zb'][:c])

    return gl, c


def _check_literal_indices(indices, **kwargs):
    """if necessary, find and replace literal indices
    """
    for k in kwargs:
        if isinstance(indices, list):  # that means it was not converted to numpy array earlier on
            indices = [kwargs[idx] if idx == k else idx for idx in indices]
        elif isinstance(indices, basestring) and k == indices:
            indices = kwargs[k]

    return indices


def read_model(restart, name):

    SEC_IN_YEAR = 3600*24*365  # conversion years <--> second

    if not os.path.exists(restart):
        raise RuntimeError('No restart file: '+restart)

    with nc.Dataset(restart, 'r') as ds:

        gl, c = _get_terminus(ds)
        #nv = sum(np.size(idx) for idx in indices) if indices is not None else len(names)  # number of variables
        #state = np.empty(nv)
        v = name

        if '?' in v:
            v, spec = v.split('?')
            idx = parse_indices(spec)
        else:
            idx = slice(None)

        idx = _check_literal_indices(idx, gl=gl, c=c)

        if v in ds.variables:
            var = ds[v][idx] if idx is not None else ds[v]

        elif v == 'F':
            var = ds['H'][idx]*ds['U'][idx]*ds['W'][idx]

        else:
            raise ValueError('unknown variable: '+v)

        if v == 'U':
            var = var*SEC_IN_YEAR

    return np.asarray(var)


def read_ensemble_loglik(outdir, constraints, runids=None, N=None, verbose=True, fmt="{:0>5}"):
    """read output state variables

    outdir: ensemble directory
    constraints: model output
    runids: sub-sample of indices to read (default: all)
    N: size of the ensemble
    fmt: format for subfolder under outdir, for each model realization
    
    OUTDIR/ID/restart.nc
    """
    if runids is None:
        if N is None:
            N = get_ensemble_size(outdir)
        runids = np.arange(N)

    N = len(runids)
    m = len(constraints)

    loglik = np.empty((N, m))
    loglik.fill(-np.inf)
    
    for i, runid in enumerate(runids):
        runid = fmt.format(i)
        out_single = os.path.join(outdir, runid)
        for j, c in enumerate(constraints):
            try:
                state_single = c.read(out_single)
            except RuntimeError as err:
                if verbose: warnings.warn("Warning::",runid,"::read::",err.message)
                continue
            try:
                loglik_single = c.logpdf(state_single)
            except RuntimeError as err:
                if verbose: warnings.warn("Warning::",runid,"::logpdf::",err.message)
                continue
            loglik[i,j] = loglik_single

    return loglik


#def _parseval(val):
#    try:
#        val = int(val)
#    except:
#        try: 
#            val = float(val)
#        except:
#            pass
#    return val


def read_model_params(outdir, pnames):
    """Read model parameter for one folder, based on "command" log (as a check)
    
    outdir : output directory for one model realization
    """
    args = open(os.path.join(outdir,'command')).read().split()
    pvalues = []
    for p in pnames:
        i = args.index('--'+p.replace('.','%'))
        val = args[i+1]
        pvalues.append(_parseval(val))
    return pvalues

def read_ensemble_params(pfile):
    pnames = open(pfile).readline().split()
    pvalues = np.loadtxt(pfile, skiprows=1)  
    return pnames, pvalues


def nans(shp):
    a = np.empty(shp)
    a.fill(np.nan)
    return a


def main():

    parser = argparse.ArgumentParser(description=__doc__, 
                                     formatter_class=RawDescriptionHelpFormatter, 
                                     epilog='Example: python costfunction.py outdir -c name=H?c type=norm?500,100 -c name=U?0:200:1 type=rms sd=20% file=glacier.nc valid=pos > weights.txt'
                                     )

    parser.add_argument('ensemble', help='ensemble directory')
    parser.add_argument('--params', help='params file, if prior is provided')
    #parser.add_argument('-o', help='output file as text format')
    #parser.add_argument('--print', action='store_true', help='print to screen')

    parser.add_argument('-c', '--constraint', nargs='*', action='append', help='Each constraint must be preceded by -c', metavar='ATTR=VAL')

    parser.add_argument('-p', '--prior', nargs='*', help='Prior Knowledge on parameter as scipy distribution', metavar='NAME=DIST?SCALE,LOC')

    #parser.add_argument('-1','-l', '--logposterior', action='store_true', help='only print final log-posterior proba')
    group = parser.add_argument_group('Input Runs')
    group.add_argument('--runids', type=int, nargs='*', help='select run ids sample')
    group.add_argument('-N', type=int, help='ensemble size - e.g. for testing, first N members')
    group.add_argument('--fmt', default='{:0>5}', help='runid to folder')

    group = parser.add_argument_group('Output')
    group.add_argument('-1','--column', action='store_true')
    group.add_argument('-s','--sep', default=" ", help='separator, when verbose is False')
    group.add_argument('-v','--verbose', action='store_true', help='verbose diagnostic including the various components of the log-proba')
    #group.add_argument('-D','--debug', action='store_true', help='debug mode, also include state variables')
    group.add_argument('--log', action='store_true', help='return log-likelihood, instead of normalized weights')

    args = parser.parse_args()

    if args.runids is None:
        if args.N is None:
            args.N = get_ensemble_size(args.ensemble)
        args.runids = np.arange(args.N)

    logprior = np.zeros(args.N, dtype=float)

    # Prior parameters?
    if args.prior:
        # read full params' file 
        if not args.params:
            args.params = os.path.join(args.ensemble, 'job.params')
        pnames, pvalues = read_ensemble_params(args.params)
        pvalues = pvalues[args.runids]

        ## read params from each output (requires 'command' file)
        #else:
        #    pnames = [p.split('=')[0] for p in args.prior]
        #    pvalues = np.empty((loglik.shape[0], len(pnames)))
        #    for i, ix in enumerate(args.runids):
        #        runid = args.fmt.format(i)
        #        out_single = os.path.join(args.ensemble, runid)
        #        pvalues[i] = read_model_params(out_single, pnames)

        n, p = pvalues.shape

        logprior_all = np.empty_like(pvalues)
        for j, par in enumerate(args.prior):
            nm, spec = par.split('=')
            dist = parse_scipydist(spec)
            try:
                par_index = pnames.index(nm)
            except:
                print("Parameters:"+repr(pnames))
                raise ValueError('Invalid parameter: '+nm)

            for i, val in enumerate(pvalues[:,par_index]):
                logprior_all[i, j] = dist.logpdf(val)

        logprior = logprior_all.sum(axis=1)


    # Compute log-likelihood
    constraints = [parse_constraint(*arg) for arg in args.constraint]
    loglik_all = read_ensemble_loglik(args.ensemble, constraints, runids=args.runids, N=args.N, fmt=args.fmt)
    loglik = loglik_all.sum(axis=1)  # resulting loglik

    logposterior = loglik + logprior

    #runids = args.runids if isinstance(args.runids, list) else np.arange(loglik.shape[0])

    
    # Prepare output
    # ==============
    if args.column:
        args.sep = '\n'

    # Standard output: just print the logposterior or weights
    if not args.verbose:

        if not args.log:
            result = np.exp(logposterior)
            result /= result.sum()
        else:
            result = logposterior
        fmt=args.sep.join(['{}']*result.size)
        print(fmt.format(*result))
        return

    # Else, print the detail of various components
    columns = ['logposterior'] 
    values = [logposterior[:,None]]

    # likelihood
    columns += ['loglik','logprior']
    values += [loglik[:,None], logprior[:,None]]

    # detail of likelihood
    columns += [c.name for c in constraints]
    values += [loglik_all]

    # detail of prior
    if args.prior:
        columns += args.prior
        values += [logprior_all]
    
    values = np.concatenate(values, axis=1)

    if not args.log:
        values = np.exp(values)
        columns = [c[3:] if c.startswith('log') else c for c in columns]

    txt = str_pmatrix(columns, values, max_rows=np.inf, include_index=True, index=args.runids)
    print(txt)

if __name__ == '__main__':
    main()

