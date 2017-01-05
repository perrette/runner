"""Read model state variables and write as a netCDF file
"""
import sys
import json
import netCDF4 as nc

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
            idx = slice(c)

        idx = _check_literal_indices(idx, gl=gl, c=c)

        if v in ds.variables:
            var = ds[v][idx] if idx is not None else ds[v]

        elif v == 'F':
            var = ds['H'][idx]*ds['U'][idx]*ds['W'][idx]*SEC_IN_YEAR*1e-9

        elif v == 'V':
            dx = ds["x"][1] - ds["x"][0]
            var = np.sum(ds['H'][idx]*ds['W'][idx]*dx)

        else:
            raise ValueError('unknown variable: '+v)

        if v == 'U':
            var = var*SEC_IN_YEAR

    return np.asarray(var)
