"""Sampling job
"""
import numpy as np
from simtools.prior import Prior, GenericParam, PriorParser
from simtools.xparams import xparams
from simtools.job.parsetools import Command, Job


# Commands
# ========

class ParamGenerator(Command):
    def __init__(self, parser):
        self.add_input_arguments(parser)
        parser.add_argument('-o', '--out', help="output parameter file")

    def __call__(self, args):
        xparams = self.generate_params(args)
        if args.out:
            with open(args.out, "w") as f:
                f.write(str(xparams))
        else:
            print(str(xparams))


class Product(ParamGenerator):
    """Factorial combination of parameter values
    """
    def add_input_arguments(self, parser):
        PriorParser.add_arguments(parser)

    def generate_params(self, args):
        prior = PriorParser.from_namespace(args)
        return prior.product()


class Sample(ParamGenerator):
    """Sample prior parameter distribution
    """
    def add_input_arguments(self, parser):

        PriorParser.add_arguments(parser)

        subp.add_argument('-N', '--size',type=int, required=True, 
                          help="Sample size")
        subp.add_argument('--seed', type=int, 
                          help="random seed, for reproducible results (default to None)")
        subp.add_argument('--method', choices=['montecarlo','lhs'], 
                          default='lhs', 
                          help="Sampling method: Monte Carlo or Latin Hypercube Sampling (default=%(default)s)")

        grp = subp.add_argument_group('Latin Hypercube Sampling (pyDOE)')
        grp.add_argument('--lhs-criterion', default=LHS_CRITERION,
                          help="see pyDOE.lhs (default=%(default)s)")
        grp.add_argument('--lhs-iterations', type=int, help="see pyDOE.lhs")


    def generate_params(self, args):

        prior = PriorParser.from_namespace(args)
        xparams = prior.sample(args.size, seed=args.seed, 
                               method=args.method,
                               criterion=args.lhs_criterion,
                               iterations=args.lhs_iterations)
        return xparams


class Resample(ParamGenerator):
    """Resample an existing parameter set using weights.
    """
    @staticmethod
    def add_weights_arguments(parser):
        group = parser.add_argument_group('weights')
        group.add_argument('-w','--weights-file', required=required)
        group.add_argument('--log', action='store_true', 
                           help='weights are provided as log-likelihood?')


    @staticmethod
    def weights_from_namespace(args):
        w = np.loadtxt(args.weights_file)
        if args.log:
            w = np.exp(log)
        return w


    def add_input_arguments(self, parser):

        #PriorParser.add_argument(parser)
        parser.add_argument("params_file", required=True,
                            help="ensemble parameter flle to resample")

        add_weights_arguments(parser)


        subp.add_argument('-N', '--size', help="New sample size (default: same size as before)", type=int)
        subp.add_argument('--seed', type=int, help="random seed, for reproducible results (default to None)")

        group = subp.add_argument_group('iis')
        group.add_argument('--iis', action='store_true', 
                          help="IIS-type resampling with likeihood flattening + jitter")
        group.add_argument('--epsilon', type=float, 
                           help='Exponent to flatten the weights and derive jitter \
    variance as a fraction of resampled parameter variance. \
            If not provided 0.05 is used as a starting value but adjusted if the \
        effective ensemble size is not in the range specified by --neff-bounds.')

        group.add_argument('--neff-bounds', nargs=2, default=NEFF_BOUNDS, type=int, 
                           help='Acceptable range for the effective ensemble size\
                           when --epsilon is not provided. Default to %(default)s.')

        group = subp.add_argument_group('sampling')
        group.add_argument('--method', choices=['residual', 'multinomial'], 
                           default=RESAMPLING_METHOD, 
                           help='resampling method (default: %(default)s)')


    def __call__(self, args):

        xpin = XParams.read(args.params_file)
        w = self.weights_from_namespace(args)

        xparams = xpin.resample(w, size=args.size, seed=args.seed,
                                method=args.method,
                                iis=args.iis, epsilon=args.epsilon, 
                                neff_bounds=args.neff_bounds, 
                                )


class Neff(Command):
    """Check effective ensemble size
    """
    def __init__(self, parser)
        Resample.add_weights_arguments(parser)
        subp.add_argument('--epsilon', type=float, default=1, 
                          help='likelihood flattening, see resample sub-command')


    def __call__(self, args):
        w = Resample.weights_from_namespace(args)
        print( Resampler(w**args.epsilon).neff() )


def main():

    job = Job()
    job.add_command("product", Product)
    job.add_command("sample", Sample)
    job.add_command("resample", Resample)
    job.add_command("neff", Neff)

    job.main()
