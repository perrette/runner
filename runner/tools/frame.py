"""Pandas-like printing
"""
def str_dataframe(pnames, pmatrix, max_rows=1e20, include_index=False, index=None):
    """Pretty-print matrix like in pandas, but using only basic python functions
    """
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


def read_dataframe(pfile):
    import numpy as np
    header = open(pfile).readline().strip()
    if header.startswith('#'):
        header = header[1:]
    pnames = header.split()
    pvalues = np.loadtxt(pfile, skiprows=1)  
    if np.ndim(pvalues) == 1:
        pvalues = pvalues[:, None]
    return pnames, pvalues


# 2-D data structure
# ==================
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
        names, values = read_dataframe(pfile)
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
        import numpy as np
        return np.arange(self.size)
