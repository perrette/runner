import argparse
import numpy as np
from runner.prior import Prior, GenericParam, DiscreteParam
import runner.sampling.resampling as xp
from runner.xparams import XParams, Resampler
from runner.register import register_job

# generate params.txt (XParams)
# =============================
def _return_params(xparams, out):
    "Return new ensemble parameters"
    if out:
        with open(out, "w") as f:
            f.write(str(xparams))
    else:
        print(str(xparams))

# product
# -------
product = argparse.ArgumentParser(add_help=False,
                                  description="Factorial combination of parameter values")
product.add_argument('factors',
                 type=DiscreteParam.parse,
                 metavar="NAME=VAL1[,VAL2 ...]",
                 nargs='*')
product.add_argument('-o','--out', help="output parameter file")


def product_post(o):
    if not o.factors:
        product.error("must provide at least one parameter")
    xparams = Prior(o.factors).product()
    return _return_params(xparams, o.out)

register_job('product', product, product_post,
                 help='generate ensemble from all parameter combinations')


# sample
# ------
prior = argparse.ArgumentParser(add_help=False)
grp = prior.add_argument_group("prior distribution of model parameters")
grp.add_argument('dist',
                 type=GenericParam.parse,
                 help=GenericParam.parse.__doc__,
                 metavar="NAME=DIST",
                 nargs='*')

lhs = argparse.ArgumentParser(add_help=False)
grp = lhs.add_argument_group("Latin hypercube sampling")
grp.add_argument('--lhs-criterion', 
                   choices=('center', 'c', 'maximin', 'm', 
                            'centermaximin', 'cm', 'correlation', 'corr'), 
                 help='randomized by default')
grp.add_argument('--lhs_iterations', type=int)


sample = argparse.ArgumentParser(description="Sample prior parameter distribution", 
                                 add_help=False, parents=[prior, lhs])
sample.add_argument('-o', '--out', help="output parameter file")

sample.add_argument('-N', '--size',type=int, 
                  help="Sample size")
sample.add_argument('--seed', type=int, 
                  help="random seed, for reproducible results (default to None)")
sample.add_argument('--method', choices=['montecarlo','lhs'], default='lhs', 
                    help="sampling method (default=%(default)s)")

def sample_post(o):
    if not o.size:
        sample.error("argument -N/--size is required")
    if not o.dist:
        sample.error("must provide at least one parameter")
    prior = Prior(o.dist)
    xparams = prior.sample(o.size, seed=o.seed, 
                           method=o.method,
                           criterion=o.lhs_criterion,
                           iterations=o.lhs_iterations)
    return _return_params(xparams, o.out)

register_job('sample', sample, sample_post,
                 help='generate ensemble by sampling prior distributions')


# resample
# --------
resample = argparse.ArgumentParser(add_help=False, description=xp.__doc__)
resample.add_argument("params_file", 
                    help="ensemble parameter flle to resample")

#grp = resample.add_argument_group('weights')
resample.add_argument('--weights-file', '-w', required=True, 
                   help='typically the likelihood from a bayesian analysis, i.e. exp(-((model - obs)**2/(2*variance), to be multiplied when several observations are used')
resample.add_argument('--log', action='store_true', 
                   help='set if weights are provided as log-likelihood (no exponential)')

grp = resample.add_argument_group('jittering')
grp.add_argument('--iis', action='store_true', 
                  help="IIS-type resampling with likelihood flattening + jitter")
grp.add_argument('--epsilon', type=float, 
                   help='Exponent to flatten the weights and derive jitter \
variance as a fraction of resampled parameter variance. \
    If not provided 0.05 is used as a starting value but adjusted if the \
effective ensemble size is not in the range specified by --neff-bounds.')

grp.add_argument('--neff-bounds', nargs=2, default=xp.NEFF_BOUNDS, type=int, 
                   help='Acceptable range for the effective ensemble size\
                   when --epsilon is not provided. Default to %(default)s.')

grp = resample.add_argument_group('sampling')
grp.add_argument('--method', choices=['residual', 'multinomial'], 
                   default=xp.RESAMPLING_METHOD, 
                   help='resampling method (default: %(default)s)')

grp.add_argument('-N', '--size', help="New sample size (default: same size as before)", type=int)
grp.add_argument('--seed', type=int, help="random seed, for reproducible results (default to None)")

grp = resample.add_argument_group('output')
grp.add_argument('-o', '--out', help="output parameter file (print to scree otherwise)")




def resample_post(o):
    weights = np.loadtxt(o.weights_file)
    if o.log:
        weights = np.exp(weights)
    if np.all(weights == 0):
        raise ValueError("all weights are zero")
    xpin = XParams.read(o.params_file)
    xparams = xpin.resample(weights, size=o.size, seed=o.seed,
                            method=o.method,
                            iis=o.iis, epsilon=o.epsilon, 
                            neff_bounds=o.neff_bounds, 
                            )
    return _return_params(xparams, o)


register_job('resample', resample, resample_post,
                 help='resample parameters from previous simulation')


# TODO : implement 1 check or tool function that returns a number of things, such as neff
## check
## -----
#def neff(argv=None):
#    """Check effective ensemble size
#    """
#    parser = CustomParser(description=neff.__doc__, parents=[], 
#                          formatter_class=argparse.RawDescriptionHelpFormatter)
#    parser.add_argument('--weights-file', '-w', required=True, 
#                       help='typically the likelihood from a bayesian analysis, i.e. exp(-((model - obs)**2/(2*variance), to be multiplied when several observations are used')
#    parser.add_argument('--log', action='store_true', 
#                       help='set if weights are provided as log-likelihood (no exponential)')
#    parser.add_argument('--epsilon', type=float, default=1, 
#                      help='likelihood flattening, see resample sub-command')
#
#    args = parser.parse_args()
#    args.weights = getweights(args.weights_file, args.log)
#
#    print( Resampler(args.weights**args.epsilon).neff() )
#
#    #job.add_command("neff", neff, 
#    #                help='(resample helper) calculate effective ensemble size')



#obs = argparse.ArgumentParser(add_help=False, description="observational constraints")
#obs.add_argument('--likelihood', '-l', dest='constraints',
#                 type=typechecker(GenericParam.parse),
#                 help=GenericParam.parse.__doc__,
#                 metavar="NAME=SPEC",
#                 nargs='*')

