"""Resample an existing parameter set

Typically, weights can be derived from a Bayesian analysis, where each
realization is compared with observations and assigned a likelihood.  An array
of resampling indices can be derived from the weights, where realizations with
large weights are resampled several times, while realization with small weights
are not resampled.  To avoid the exact same parameter set to appear duplicated
in the resampled ensemble, introduction of noise (jitter) is necessary, which
conserves statistical properties of the resampled ensemble (covariance).

The problem is not trivial and several approaches exist for both the sampling
of indices and the addition of noise. Basically, differences in resampling
methods (before application of jitter) mainly affect how the tail - low-weights
realizations - are dealt with, which influences the results for "small"
ensemble size:

- multinomial : random sampling based on empirical distribution function.
    Simple but poor performance.
- residual : some of the resampling indices can be determined deterministically 
    when weights are large enough, i.e. `w_i * N > 1` where `w_i` represents 
    a normalized weight (sum of all weights equals 1), and N is the ensemble size.
    The array of weight residuals (`w_i * N - int(w_i * N)`) is then resampled
    using a basic multinomial approach.

More advanced methods are typically similar to `residual`, but the array of
residual weights is resampled taking into account the uniformity of samples in
the parameter or state space (and therefore requires additional information).
One of these methods, coined `deterministic` (re)sampling, is planned to be
implemented, in addition to the two mentioned above.

The jittering step is tricky because the noise is unlikely to have a pure
(multivariate) normal distribution (especially when the model is strongly non
linear).  An approach proposed by Annan and Heargraves, "iterative importance
sampling" (`iis`), is to sample jitter with zero mean and covariance computed from the
original (resampled) ensemble but scaled so that its variance is only a small
fraction `epsilon` of the original ensemble. Addition of noise increases
overall covariance by `1 + epsilon`, but they show that this can balance out if
the weights used for resampling are "flattened" with the same `epsilon` as an
exponent (`shrinking`).  This procedure leaves the posterior distribution
invariant, so that it can be applied iteratively when starting from a prior
which is far from the posterior. 

One step of this resampling procedure can be activated with the `--iis` flag.
By default the epsilon factor is computed automatically to keep an "effective
ensemble size" in a reasonable proportion (50% to 90%) to the actual ensemble
size (see `--neff-bounds` parameter). No other jittering method is proposed.
"""
import numpy as np
import argparse
from simtools.parsetools import CustomParser
import simtools.sampling.resampling as xp
from simtools.xparams import XParams

def getweights(weights_file, log=False):
    w = np.loadtxt(weights_file)
    if log:
        w = np.exp(log)
    return w


def resample_main(argv=None):
    """Resample an existing ensemble set using weights.
    """
    parser = CustomParser(description=__doc__, parents=[], 
                          formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("params_file", 
                        help="ensemble parameter flle to resample")

    group = parser.add_argument_group('weights')
    group.add_argument('--weights-file', required=True, 
                       help='typically the likelihood from a bayesian analysis, i.e. exp(-((model - obs)**2/(2*variance), to be multiplied when several observations are used')
    group.add_argument('--log', action='store_true', 
                       help='set if weights are provided as log-likelihood (no exponential)')

    group = parser.add_argument_group('jittering')
    group.add_argument('--iis', action='store_true', 
                      help="IIS-type resampling with likelihood flattening + jitter")
    group.add_argument('--epsilon', type=float, 
                       help='Exponent to flatten the weights and derive jitter \
variance as a fraction of resampled parameter variance. \
        If not provided 0.05 is used as a starting value but adjusted if the \
    effective ensemble size is not in the range specified by --neff-bounds.')

    group.add_argument('--neff-bounds', nargs=2, default=xp.NEFF_BOUNDS, type=int, 
                       help='Acceptable range for the effective ensemble size\
                       when --epsilon is not provided. Default to %(default)s.')

    group = parser.add_argument_group('sampling')
    group.add_argument('--method', choices=['residual', 'multinomial'], 
                       default=xp.RESAMPLING_METHOD, 
                       help='resampling method (default: %(default)s)')

    group.add_argument('-N', '--size', help="New sample size (default: same size as before)", type=int)
    group.add_argument('--seed', type=int, help="random seed, for reproducible results (default to None)")


    # output
    group = parser.add_argument_group('output')
    group.add_argument('-o', '--out', help="output parameter file (print to scree otherwise)")


    o = parser.parse_args(argv)
    #o = parser.postprocess(o)
    o.weights = getweights(o.weights_file, o.log)

    xpin = XParams.read(o.params_file)
    xparams = xpin.resample(o.weights, size=o.size, seed=o.seed,
                            method=o.method,
                            iis=o.iis, epsilon=o.epsilon, 
                            neff_bounds=o.neff_bounds, 
                            )
    return return_params(xparams, o)
