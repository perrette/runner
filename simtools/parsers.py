"""Interface for prior parameter
"""
import argparse
import os
import json

from simtools.prior import Prior, GenericParam, PRIOR_KEY

# Compared to ArgumentParser, this object return an initialized class
class PriorParser(object):
    """Prior param
    """
    def __init__(self, parser=None, **kwargs):
        self.parser = parser or argparse.ArgumentParser(description=self.__doc__, **kwargs)

        grp = self.parser.add_argument_group("prior parameters")
        x = grp.add_mutually_exclude_arguments()
        x.add_argument('-p', '--prior-params', default=[], nargs='*', 
                                type=GenericParam, metavar="NAME=SPEC", 
                                help="Prior parameter defintion. \
SPEC specifies a param values or distribution (depending on the sub-command).\
            Discrete parameter values can be provided \
            as a comma-separated list `VALUE[,VALUE...]` \
            or a range `START:STOP:N`. \
            A distribution is provided as `TYPE?ARG,ARG[,ARG,...]`. \
            Pre-defined `U?min,max` (uniform) and `N?mean,sd` (normal) \
            or any scipy.stats distribution as TYPE?[SHP,]LOC,SCALE.")

        x.add_argument('--prior-file', 
                         help='prior parameter file (json file with "'+PRIOR_KEY+'" key)')

        grp.add_argument('--prior-key', default=PRIOR_KEY, help=argparse.SUPPRESS)

        x = grp.add_mutually_exclude_arguments()
        x.add_argument('--only-params', nargs='*', 
                         help="filter out all but these parameters")
        x.add_argument('--exclude-params', nargs='*', 
                         help="filter out these parameters")

    
    @staticmethod
    def get_prior(args):
        """return Prior class
        """
        if args.prior_file:
            prior = Prior.read(args.prior_file, args.prior_key)
            if args.only_params
                prior.filter_params(args.only_params, keep=True)
            if args.exclude_params:
                prior.filter_params(args.exclude_params, keep=False)

        else:
            prior = Prior(args.prior_params)

        return prior


    def parse_args(self, argv=None)
        args = self.parser.parse_args(argv)
        return self.get_prior(args)


    def parse_known_args(self, argv=None)
        args, unknown = self.parser.parser_known_args(argv)
        return self.get_prior(args), unknown

    @staticmethod
    def get_prior(args):
        """Return Prior instance
        """
        if not args.prior_file and os.path.exists(DEFAULT_CONFIG):
            args.prior_file = DEFAULT_CONFIG

        if args.prior_file:
            prior = Prior.read(args.prior_file)
            update_params(prior.params, args.prior_params)
        else:
            prior = Prior(args.prior_params)
        return prior






# for the sake of the scripts...
class ParamsParser(object):
    """Helper class to build ArgumentParser with subcommand and keep clean
    """
    def __init__(self, *args, **kwargs):
        self.parser = argparse.ArgumentParser(*args, **kwargs)
        self.subparsers = self.parser.add_subparsers(dest='cmd')
        self.define_parents()

    def define_parents(self):
        " arguments shared in various commands "
        # input prior parameters
        self.prior = argparse.ArgumentParser(add_help=False)
        grp = self.prior.add_argument_group("prior parameters")
        grp.add_argument('-p', '--prior-params', default=[], nargs='*', 
                                type=parse_param, metavar="NAME=SPEC", 
                                help="Prior parameter defintion. \
SPEC specifies a param values or distribution (depending on the sub-command).\
            Discrete parameter values can be provided \
            as a comma-separated list `VALUE[,VALUE...]` \
            or a range `START:STOP:N`. \
            A distribution is provided as `TYPE?ARG,ARG[,ARG,...]`. \
            Pre-defined `U?min,max` (uniform) and `N?mean,sd` (normal) \
            or any scipy.stats distribution as TYPE?[SHP,]LOC,SCALE.")

        grp.add_argument('--prior-file', help='experimental')

        ## size & sample
        #self.size = argparse.ArgumentParser(add_help=False)

        # input param file
        self.pin = argparse.ArgumentParser(add_help=False)
        self.pin.add_argument('params-file', help='input parameter file')

        # input ensemble weights
        self.win = argparse.ArgumentParser(add_help=False)
        group = self.win.add_argument_group('weights')
        group.add_argument('-w','--weights-file', required=True)
        group.add_argument('--log', action='store_true', 
                           help='weights are provided as log-likelihood?')

        # output param file
        self.pout = argparse.ArgumentParser(add_help=False)
        self.pout.add_argument('-o', '--out', help="output parameter file")


    def add_product(self):
        """factorial combination of parameter values
        """
        subp = self.subparsers.add_parser("product", parents=[self.prior, self.pout],
                                     help=self.add_product.__doc__)
        return subp

    def add_sample(self):
        """Sample prior parameter distribution
        """
        subp = self.subparsers.add_parser("sample", parents=[self.prior, self.pout], 
                                     help=__doc__)

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
        return subp


    def add_resample(self):
        """Resample an existing parameter set using weights.
        """
        subp = self.subparsers.add_parser("resample", 
                                          parents=[self.pout, self.pin, self.win], 
                                     help=__doc__)

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
        return subp

    def add_neff(self):
        """Check effective ensemble size
        """
        subp = self.subparsers.add_parser("neff", parents=[self.pin, self.win], 
                                     help=__doc__)
        subp.add_argument('--epsilon', type=float, default=1, 
                          help='likelihood flattening, see resample sub-command')

        return subp
        #group.add_argument('--neff-only', action='store_true', help='effective ensemble size')

    def parse_args(self, *args, **kwargs):
        return self.parser.parse_args(*args, **kwargs)



def main(argv=None):

    parser = ParamsParser(description=__doc__,
            epilog='Examples: \n ./genparams.py product -p a=0,2 b=0:3:1 c=4 \n ./genparams.py sample -p a=uniform?0,10 b=norm?0,2 --method lhs --size 4',
            formatter_class=argparse.RawDescriptionHelpFormatter)

    # add subcommands
    parser.add_product()
    parser.add_sample()
    parser.add_resample()
    parser.add_neff()

    args = parser.parse_args(argv)

    prior = get_prior(args)

    # Combine parameter values
    # ...factorial model: no numpy distribution allowed
    if args.cmd == 'product':
        xparams = prior.product()

    # ...monte carlo and lhs mode
    elif args.cmd == 'sample':
        xparams = prior.sample(args.size, seed=args.seed, 
                               method=args.method,
                               criterion=args.lhs_criterion,
                               iterations=args.lhs_iterations)

    elif args.cmd == 'resample':

        xpin = XParams.read(args.params_file)
        w = np.loadtxt(args.weights_file)
        if args.log:
            w = np.exp(log)
        xparams = xpin.resample(w, size=args.size, seed=args.seed,
                                method=args.method,
                                iis=args.iis, epsilon=args.epsilon, 
                                neff_bounds=args.neff_bounds, 
                                )

    elif args.cmd == 'neff':
        w = np.loadtxt(args.weights_file)
        if args.log:
            w = np.exp(log * args.epsilon)
        else:
            w = w ** args.epsilon
        print( Resampler(w).neff() )
        return


    if args.out:
        with open(args.out,'w') as f:
            f.write(str(xparams))
    else:
        print (str(xparams))
