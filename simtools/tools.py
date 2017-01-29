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

    def scatter_matrix(self, **kwargs):
        " call to pandas.scatter_matrix "
        import pandas as pd
        return pd.scatter_matrix(self.df, **kwargs)

    def parallel_coordinates(self, name=None, colormap=None, alpha=0.5, 
                             add_cb=True, cb_axes=[0.05, 0, 0.9, 0.05], 
                             normalize=True, 
                             cb_orientation='horizontal', **kwargs):
        """Call to pandas.parallel_coordinates + customization
        
        * name : variable name along which to sort values
        * normalize : True by default
        * colormap : e.g. viridis, inferno, plasma, magma
		    http://matplotlib.org/examples/color/colormaps_reference.html
        * add_cb : add a colorbar
		    http://matplotlib.org/examples/api/colorbar_only.html
        * cb_axes : tuned for horizontal cb with 5 variables
        * cb_orientation : horizontal or vertical
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import pandas as pd
        df = self.df.dropna()
        if normalize:
            delta = (df - df.mean()) / df.std()  # normalize
        else:
            delta = df

        if name is None:
            name = self.names[0]

        # insert class variable as new variable
        cls = df[[name]].rename(columns={name:'cls'})
        full = pd.concat([cls, delta], axis=1).sort_values('cls')

        # http://matplotlib.org/examples/api/colorbar_only.html
        # http://matplotlib.org/examples/color/colormaps_reference.html
        cmap = colormap or mpl.cm.viridis
        axes = pd.tools.plotting.parallel_coordinates(full, 'cls', alpha=alpha, 
                                                      colormap=cmap, ax=ax)
        axes.legend().remove()  # remove the legend
        axes.set_ylabel('normalized')

        # add colorbar axis (since no mappable is easily found...)
        if not add_cb:
            return axes

        fig = plt.gcf()
        if add_cb:
            ax1 = fig.add_axes(cb_axes)
        norm = mpl.colors.Normalize(vmin=cls.values.min(), vmax=cls.values.max())
        cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                        norm=norm, 
                                        orientation=cb_orientation)
        cb1.set_label(name)
        return axes

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
