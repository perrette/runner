"""distribution I/O
"""
import numpy as np
from runner.tools.misc import parse_val

class LazyDist(object):
    " lazy loading of scipy distributions "
    def __init__(self, name):
        self.name = name

    def __call__(self, *args, **kwargs):
        import scipy.stats.distributions
        dist = getattr(scipy.stats.distributions, self.name)
        return dist(*args, **kwargs)

norm = LazyDist('norm')
uniform = LazyDist('uniform')
rv_continuous = LazyDist('rv_continuous')
rv_discrete = LazyDist('rv_discrete')
rv_frozen = LazyDist('rv_frozen')


def dist_todict(dist):
    """scipy dist to keywords
    """
    dist_gen = dist.dist
    n = len(dist_gen.shapes.split()) if dist_gen.shapes else 0
    shapes = dist.args[:n]
    kw = {'name': dist_gen.name, 'loc':0, 'scale':1}
    kw.update(dist.kwds)
    if shapes:
        kw['shapes'] = shapes
    assert len(dist.args[n:]) <= 2, dist.name
    if len(dist.args[n:]) >= 1:
        kw['loc'] = dist.args[n]
    if len(dist.args[n:]) == 2:
        kw['scale'] = dist.args[n+1]
    return kw


def dist_fromkw(name, **kwargs):
    """scipy dist to keywords
    """
    import scipy.stats.distributions as mod
    dist = getattr(mod, name)
    args = list(kwargs.pop('shapes', [])) + [kwargs.pop('loc',0), kwargs.pop('scale',1)]
    assert not kwargs, name
    return dist(*args)


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
    """List of parameters: VALUE[,VALUE,...]
    """
    if not string:
        raise ValueError("empty list")
    return [parse_val(value) for value in string.split(',')]

def parse_range(string):
    """Parameter range: START:STOP:N
    """
    start, stop, n = string.split(':')
    start = float(start)
    stop = float(stop)
    n = int(n)
    return np.linspace(start, stop, n).tolist()

def parse_dist(string):
    """Distribution:

    N?MEAN,STD or U?MIN,MAX or TYPE?ARG1[,ARG2 ...] 
    where TYPE is any scipy.stats distribution with *shp, loc, scale parameters.
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
        dist = LazyDist(name)(*args)

    return dist

#from runner.backends.dist import parse_dist, dist_to_str, LazyDist, dist_fromkw
class DiscreteDist(object):
    """Prior parameter that takes a number of discrete values
    """
    def __init__(self, values):
        self.values = np.asarray(values)

    def rvs(self, size):
        indices = np.random.randint(0, len(self.values), size)
        return self.values[indices]

    def ppf(self, q, interpolation='nearest'):
        return np.percentile(self.values, q*100, interpolation=interpolation)

    def __str__(self):
        return ",".join(*[str(v) for v in self.values])

    @classmethod
    def parse(cls, string):
        if ':' in string:
            values = parse_range(string)
        else:
            values = parse_list(string)
        return cls(values)


def parse_dist2(string):
    if '?' in string:
        return parse_dist(string)
    else:
        return DiscreteDist.parse(string)

def dist_to_str2(dist):
    if isinstance(dist, DiscreteDist):
        return str(dist)
    else:
        return dist_to_str(dist)

def dist_todict2(dist):
    if isinstance(dist, DiscreteDist):
        return {'values':dist.values.tolist(), 'name':'discrete'}
    return dist_todict(dist)

def dist_fromkw2(name, **kwargs):
    if name == 'discrete':
        return DiscreteDist(**kwargs)
    return dist_fromkw(name, **kwargs)




def cost(dist, value):
    " logpdf = -0.5*cost + cte, only makes sense for normal distributions "
    logpdf = dist.logpdf(value)
    cst = dist.logpdf(dist.mean())
    return -2*(logpdf - cst)


def dummydist(default):
    """dummy distribution built on rv_continuous

    Example
    -------
    >>> dummy = dummydist(3)
    >>> dummy.interval(0.9)
    (-inf, inf)
    >>> dummy.pdf(0)
    1.0
    >>> dummy.logpdf(0)
    0.0
    >>> dummy.rvs(2)
    np.array([3.0, 3.0])
    """
    from scipy.stats import rv_continuous
    class dummy_gen(rv_continuous): 
        def _pdf(self, x):
            return 1
        def _ppf(self, x): # for interval to work
            return np.inf if x >= 0.5 else -np.inf
        def rvs(self, size=None, loc=0, **kwargs):
            return np.zeros(size)+loc if size is not None else loc
    dummy = dummy_gen('none')
    return dummy(loc=default)

