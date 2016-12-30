#!/usr/bin/env python2.7
"""Generate parameter ensemble
"""
from __future__ import print_function
import argparse, os
from argparse import RawDescriptionHelpFormatter
from itertools import product
#import numpy as np

# parse parameters from command-line
# ==================================
def _parse_val(s):
    " string to int, float, str "
    try:
        val = int(s)
    except:
        try:
            val = float(s)
        except:
            val = s
    return val

def parse_param_list(string):
    """Parse list of parameters VALUE[,VALUE,...]
    """
    return [_parse_val(value) for value in string.split(',')]

def parse_param_range(string):
    """Parse parameters START:STOP:STEP
    """
    import numpy as np
    return np.arange(*[_parse_val(value) for value in string.split(':')]).tolist()

def parse_param_dist(string):
    """Parse parameters dist?loc,scale
    """
    import scipy.stats.distributions as sd
    name,params = string.split('?')
    loc, scale = params.split(',')
    return getattr(sd,name)(_parse_val(loc), _parse_val(scale))


def params_parser(string):
    """used as type by ArgumentParser
    """
    try:
        name, spec = string.split('=')
        if '?' in spec:
            params = parse_param_dist(spec)
        elif ':' in spec:
            params = parse_param_range(spec)
        else:
            params = parse_param_list(spec)
    except Exception as error:
        print( "ERROR:",error.message)
        raise
    return name,params


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
            epilog='Examples: \n ./genparams.py -p a=0,2 b=0:3:1 c=4 \n ./genparams.py -p a=uniform?0,10 b=norm?0,2 --mode lhs -N 4',
            formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument('-p', '--params', default=[], type=params_parser, nargs='*', metavar="NAME=SPEC",
            help="Modified parameters. SPEC is one of {list: VALUE[,VALUE...] OR range: START:STOP:STEP OR scipy dist NAME?LOC,SCALE e.g. norm?mean,sd or uniform?min,max}")

    parser.add_argument('--mode', choices=['factorial','montecarlo','lhs'], default='factorial', help="Sampling mode: factorial, Monte Carlo or Latin Hypercube Sampling")
    parser.add_argument('--lhs-criterion', help="pyDOE lhs parameter")
    parser.add_argument('--lhs-iterations', help="pyDOE lhs parameter")
    parser.add_argument('-i','--from-file', help="look in file for any parameter provided as params, and use instead of command-line specification")
    parser.add_argument('-N', '--size', help="Sample size (montecarlo or lhs modes)", default=100, type=int)
    parser.add_argument('-o', '--out', help="Output parameter file")
    parser.add_argument('--seed', type=int, help="random seed, for reproducible results (default to None)")

    args = parser.parse_args()

    pnames = [nm for nm, vals in args.params]

    # Combine parameter values
    # ...factorial model: no numpy distribution allowed
    if args.mode == 'factorial':
        for nm, p in args.params:
            if not isinstance(p, list):
                raise ValueError('genparams.py: only list and ranges allowed in factorial mode. Got: '+repr(p))
        pvalues = list(product(*[vals for nm, vals in args.params]))

    # ...monte carlo and lhs mode
    elif args.mode == 'lhs':

        import numpy as np
        pvalues = np.empty((args.size,len(pnames)))

        from pyDOE import lhs
        np.random.seed(args.seed)
        lhd = lhs(len(pnames), args.size, args.lhs_criterion, args.lhs_iterations) # sample x parameters, all in [0, 1]

        for i, pp in enumerate(args.params):
            nm, spec = pp
            if isinstance(spec, list):
                pvalues[:,i] = spec
            else:
                pvalues[:,i] = spec.ppf(lhd[:,i]) # take the percentiles for the particular distribution

    # ...montecarlo
    else:
        import numpy as np
        pvalues = np.empty((args.size,len(pnames)))

        for i, pp in enumerate(args.params):
            nm, spec = pp
            if isinstance(spec, list):
                pvalues[:,i] = spec
            else:
                pvalues[:,i] = spec.rvs(size=args.size, random_state=args.seed) # scipy distribution: sample !

    # in case a file is provided as input, just use values from files
    if args.from_file:
        pnames0 = open(args.from_file).readline().split()
        pvalues0 = np.loadtxt(args.from_file, skiprows=1)
        for i, nm in enumerate(pnames):
            if nm in pnames0:
                #print('Overwrite {} with values from {}'.format(nm, args.from_file))
                pvalues[:,i] = pvalues0[:, pnames0.index(nm)]

    params_str = str_pmatrix(pnames, pvalues, include_index=False)
    if args.out:
        with open(args.out,'w') as f:
            f.write(params_str)
    else:
        print (params_str)
