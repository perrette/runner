"""Tools
"""
import numpy as np
from scipy.stats import norm, uniform
import scipy.stats.distributions

def parse_val(s):
    " string to int, float, str "
    try:
        val = int(s)
    except:
        try:
            val = float(s)
        except:
            val = s
    return val

#def parse_keyval(string):
#    name, value = string.split("=")
#    value = parse_val(value)
#    return name, value


def nans(N):
    a = np.empty(N)
    a.fill(np.nan)
    return a


# Sscipy Dist String I/O (useful for command line)
# ======================

# param to string
# ---------------
def dist_to_str(dist):
    """format scipy-dist distribution
    """
    dname=dist.dist.name
    dargs=dist.args

    # hack (shorted notation)
    dname = dname.replace("norm","N")
    if dname == "uniform":
        dname = "U"
        loc, scale = dargs
        dargs = loc, loc+scale  # more natural

    sargs=",".join([str(v) for v in dargs])
    return "{}?{}".format(dname, sargs)


# string to param
# ---------------
def parse_list(string):
    """Parse list of parameters VALUE[,VALUE,...]
    """
    if not string:
        raise ValueError("empty list")
    return [parse_val(value) for value in string.split(',')]

def parse_range(string):
    """Parse parameters START:STOP:N
    """
    start, stop, n = string.split(':')
    start = float(start)
    stop = float(stop)
    n = int(n)
    return np.linspace(start, stop, n).tolist()

def parse_dist(string):
    """Parse distribution dist?loc,scale
    """
    name,spec = string.split('?')
    args = [float(a) for a in spec.split(',')]
    
    # alias for common cases
    if name == "N":
        mean, std = args
        dist = norm(mean, std)

    elif name == "U":
        lo, hi = args  # note: uniform?loc,scale differs !
        dist = uniform(lo, hi-lo) 

    else:
        dist = getattr(scipy.stats.distributions, name)(*args)

    return dist


# 2-D data structure
# ==================


def str_dataframe(pnames, pmatrix, max_rows=1e20, include_index=False, index=None):
    """Pretty-print matrix like in pandas, but using only basic python functions
    """
    #assert isinstance(pnames[0], basestring), type(pnames[0])
    #assert isinstance(pmatrix[0][0], float), type(pmatrix[0][0])
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


def read_df(pfile):
    import numpy as np
    header = open(pfile).readline().strip()
    if header.startswith('#'):
        header = header[1:]
    pnames = header.split()
    pvalues = np.loadtxt(pfile, skiprows=1)  
    return pnames, pvalues


class DataFrame(object):
    """DataFrame with names and matrix : Parameters, State variable etc
    """
    def __init__(self, values, names):
        self.values = values
        self.names = names

    @property
    def df(self):
        " convert to pandas dataframe "
        import pandas as pd
        return pd.DataFrame(self.values, columns=self.names)

    @property
    def plot(self):
        " convert to pandas dataframe "
        return self.df.plot

    @classmethod 
    def read(cls, pfile):
        names, values = read_df(pfile)
        return cls(values, names)

    def write(self, pfile):
        with open(pfile, "w") as f:
            f.write(str(self))

    # make it like a pandas DataFrame
    def __getitem__(self, k):
        return self.values[:, self.names.index(k)]

    def keys(self):
        return self.names

    def __str__(self):
        return str_dataframe(self.names, self.values, index=self.index)

    @property
    def size(self):
        return len(self.values)

    def __iter__(self):
        for k in self.names:
            yield k

    @property
    def __len__(self):
        return self.values.shape[1]

    @property
    def shape(self):
        return self.values.shape

    @property
    def index(self):
        return np.arange(self.size)



def _create_dirtree(a,chunksize=2):
    """create a directory tree from a single, long name

    e.g. "12345" --> ["1", "23", "45"]
    """
    b = a[::-1]  # reverse
    i = 0
    l = []
    while i < len(b):
        l.append(b[i:i+chunksize])
        i += chunksize
    return [e[::-1] for e in l[::-1]]
