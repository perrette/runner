"""Glacier-model-specific methods (set params, read output)
"""
import os, sys
import json, warnings
import netCDF4 as nc
import numpy as np
from collections import OrderedDict as odict

GLACIER = "glacier"
SEC_IN_YEAR = 3600*24*365.25

# simulation: set parameters
# --------------------------
def maybe_transform_param(name, value):
    """return param name and value understood by actual model
    """
    return name, value


def driving_stress(x, z, h, rho_i=910, g=9.81):
    return -rho_i*g*h[:-1]*(z[1:]-z[:-1])/(x[1:]-x[:-1])


def autoset_params(netcdf):
    import netCDF4 as nc
    ds = nc.Dataset(netcdf)

    tauc_max = driving_stress(ds["x"][:2], ds["surf"][:2], ds["H"][:2])[0]*1e-3
    assert tauc_max > 0, 'driving stress < 0, cant autoset sliding params, maybe smooth?'
    uq = ds["U"][0]*SEC_IN_YEAR
    h0 = ds["H"][0]

    ds.close()
    return tauc_max, uq, h0


def glacierargs(netcdf, outdir, *args, **kwargs):
    """Return glacier command (to be executed from the shell)

    Parameters
    ----------
    netcdf: input glacier netcdf
    outdir: output directory
    *args: variable list of parameters
    **kwargs: keyword parameters

    Returns
    -------
    cmd : list of parameters starting with executable (to obtain a string: " ".join(cmd))
    """
    # data-dependent parameters
    params = odict()
    tauc_max, uq, h0 = autoset_params(netcdf)
    params["dynamics%tauc_max"] = tauc_max
    params["dynamics%Uq"] = uq
    params["dynamics%H0"] = h0
    for k in sorted(kwargs.keys()):
        params[k] = kwargs[k]

    # make command line argument for glacier executable
    cmd = ["--in_file", netcdf, "--out_dir", outdir] + [str(a) for a in args]
    for k in params:
        name, value = maybe_transform_param(k, params[k])
        cmd.append("--"+name)
        cmd.append(str(value))

    #cmdstr = " ".join(cmd) + (" " + cmd_extra if cmd_extra else "")
    return [GLACIER] + cmd


# analyze output, define custom variables
# ---------------------------------------
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


def _check_literal_indices(indices, **kwargs):
    """if necessary, find and replace literal indices
    """
    for k in kwargs:
        if isinstance(indices, list):  # that means it was not converted to numpy array earlier on
            indices = [kwargs[idx] if idx == k else idx for idx in indices]
        elif isinstance(indices, basestring) and k == indices:
            indices = kwargs[k]

    return indices


def read_model(ds, name):

    SEC_IN_YEAR = 3600*24*365  # conversion years <--> second

    if isinstance(ds, basestring):
        restart = ds
        if not os.path.exists(restart):
            raise RuntimeError('No restart file: '+restart)
        ds = nc.Dataset(restart)
    else:
        restart = None

    #with nc.Dataset(restart, 'r') as ds:

    gl, c = _get_terminus(ds)
    #nv = sum(np.size(idx) for idx in indices) if indices is not None else len(names)  # number of variables
    #state = np.empty(nv)
    v = name

    if '?' in v:
        v, spec = v.split('?')
        idx = parse_indices(spec)
    else:
        idx = slice(c)

    idx = _check_literal_indices(idx, gl=gl, c=c)

    if v in ds.variables:
        var = ds[v][idx] if idx is not None else ds[v]

    elif v == 'F':
        var = ds['H'][idx]*ds['U'][idx]*ds['W'][idx]*SEC_IN_YEAR*1e-9

    elif v == 'V':
        dx = ds["x"][1] - ds["x"][0]
        var = np.sum(ds['H'][idx]*ds['W'][idx]*dx)

    elif v == 'smb_mean':
        w = ds['W'][:gl]
        var = np.sum(ds['smb'][:gl]*w) / np.sum(w) * SEC_IN_YEAR

    else:
        raise ValueError('unknown variable: '+v)

    if v == 'U':
        var = var*SEC_IN_YEAR

    # close dataset if needed
    if restart is not None:
        ds.close()

    if np.iterable(var):
        var = np.asarray(var)

    return var


# define constraints

def nans(shp):
    a = np.empty(shp)
    a.fill(np.nan)
    return a
