"""Core runner code (see README.md)
"""
from __future__ import absolute_import
import os, sys, shutil, copy
import argparse
from argparse import RawDescriptionHelpFormatter
from itertools import groupby, chain

from .tools import run_foreground, run_background, submit_job, ask_user
from .sampling import combiner
from .parameters import Param, Params



# Template for Model class
# ========================
class Model(object):
    """Class to wrap an executable.

    Attributes
    ----------
    params: `Params` instance 
        list of modifiable parameters for the model

    Methods
    -------
    setup_outdir: setup output directory and returns executable & command-line args
    update_params: by default, params' update method is used
    """
    executable = None
    cmd_args = ""
    params_cls = None   # designed

    def __init__(self, params_file):
        """Initialize model from a parameter file.
        
        This method should be subclassed.
        Below some typical implementation designed to be general
        """
        # check some typical file format...
        if self.params_cls is None:
            if params_file.endswith('.nml'):
                from .namelist import Namelist
                params_cls = Namelist
            elif params_file.endswith('.json'):
                params_cls = Params  
            else:
                params_cls = Params
        else:
            params_cls = Params

        self.params = params_cls(params_file)
        self.params_file = os.path.basename(params_file)  # keep that in memory


    def update_params(self, params):
        self.params.update(params) # params can be a subset or all of model parameters


    def setup_outdir(self, outdir):
        """Setup model to run in the provided output directory

        Parameters
        ----------
        outdir : output directory for the model

        Returns
        -------
        exe, cmd_args : executable and command-line arguments' string
        """
        assert self.executable is not None, "need to set executable !"

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # Write parameters to disk
        self.params.write(os.path.join(outdir, self.params_file))

        return os.path.abspath(self.executable), self.cmd_args.format(outdir=outdir)


    def run(self, out_dir, background=False, ini_dir=None):
        """Run model locally (terminal or background)
        """
        exe, cmd_args = self.setup_outdir(out_dir)

        if background:
            job_id = run_background(exe, cmd_args, ini_dir, out_dir)
        else:
            job_id = run_foreground(exe, cmd_args, ini_dir)
        return job_id


    def submit(self, out_dir, **job_args):
        """Run model locally (terminal or background)
        """
        exe, cmd_args = self.setup_outdir(out_dir)
        return submit_job(exe, cmd_args, out_dir, **job_args)


# Match string parameter name and actual model parameter
# ------------------------------------------------------

def parse_param_name(string):
    """input_string : [MODULE:][BLOCK&]NAME
    parse and return name, block, module ; None if not provided

    where BLOCK and MODULE, if not provided, will be guessed from model's default parameters.
    To each MODULE corresponds a set of default parameters (and a parameter file).
    Note BLOCK is namelist-specific, and may be deprecated in future versions.
    """
    name = string
    if ':' in name:
        module, name = name.split(':')
    else:
        module = None
    if '&' in name:
        block, name = name.split("&")
    else:
        block = None
    return name, block, module


def lookup_param(pname, params, default_module=None, default_block=None):
    """Lookup matching default parameter among all model parameters

    Parameters
    ----------
    pname : str
        parameter name as [MODULE:][BLOCK&]NAME
    params : Params' instance
        default Model's parameters

    default_module : str, optional
        default value for Param's `module` attribute
    default_block : str, optional
        default value for Param's `block` attribute

    Returns
    -------
    corresponding default Param instance in model
    """
    # initiate a Param instance from parameter name
    name, block, module = parse_param_name(pname)
    if module is None: module = default_module
    if block is None: block = default_block

    p = Param(name=name, group=block, module=module)

    import difflib
    matches = [par for par in params if par==p]
    if not matches:
        print sys.argv[0], ":: error :: invalid parameter:",repr(p.key)
        suggestions = difflib.get_close_matches(p.name, [par.name for par in params])
        print "Do you mean: ", ", ".join(suggestions), "?"
        sys.exit(-1)
    elif len(matches) > 1:
        print sys.argv[0], ":: error :: several matches for param", repr(p.key)
        print "Matches:",matches
        print "Please specify module or block as [MODULE:][BLOCK&]NAME=VALUE[,VALUE...] or via --module and --block parameters"
        sys.exit(-1)
    assert len(matches) == 1
    return matches[0]



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


def parse_param_factors(string):
    """modified parameters as NAME=VALUE[,VALUE,...]

    Returns
    -------
    name : str
    values : list of values (int, float, str)
    """
    name, value = string.split('=')
    if ',' in value:
        values = value.split(',')
    else:
        values = [value]
    return name, [_parse_val(value) for value in values]


def params_parser(string):
    """used as type by ArgumentParser
    """
    try:
        params = parse_param_factors(string)
    except Exception as error:
        print "ERROR:",error.message
        raise
    return params


# Parameters ensemble
# ===================
# Note that Param / Params classes are more for I/O with the model
# while parameters ensemble uses python object to handle parameter combinations internally.

# String representation of parameter ensembles
# --------------------------------------------
def str_pmatrix(pnames, pmatrix, max_rows=10, include_index=True):
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
            pset = [i] + list(pset)
        lines.append(line_fmt.format(*pset))

    n = len(lines)
    # full print
    if n <= max_rows:
        return "\n".join([header]+lines)

    # partial print
    else:
        sep = line_fmt.format(*['.'*min(3,w) for w in col_width])  # separator '...'
        return "\n".join([header]+lines[:max_rows/2]+[sep]+lines[-max_rows/2:])


def str_factors(factors):
    """
    factors is a list of (pname, pvalues),  not yet combined
    e.g. [("a", [1, 2, 3]), ("b", [1.1, 2.2, 3.3])]
    """
    lines = []
    for nm, values in factors:
        lines.append(" {} = {}".format(nm, ", ".join([repr(v) for v in values])))
    return "\n".join(lines)


# Parameters ensemble I/O
# -----------------------
def read_params_file(params_file, dtype=None):
    """read parameters matrix (from previous ensemble)
    dtype: parameters' type (by default: None means guessed from file)
    """
    import numpy as np
    pmatrix = np.genfromtxt(params_file, names=True, dtype=dtype)
    pnames = pmatrix.dtype.names
    return pnames, pmatrix.tolist()


def write_params_file(params_file, pnames, pmatrix):
    """write parameters matrix
    """
    txt = str_pmatrix(pnames, pmatrix, max_rows=len(pmatrix), include_index=False)
    with open(params_file, 'w') as f:
        f.write(txt)
    # import numpy as np
    # np.savetxt(params_file, pmatrix, header=" ".join(pnames), comments="")


# class Ensemble(object):
#     """ensemble of parameters
#     """
#     def __init__(self, pnames, pmatrix):
#         self.pnames = pnames
#         self.pmatrix = pmatrix
#
#     def write(self, params_file):
#         write_params_file(params_file, self.pnames, self.pmatrix)
#
#     @classmethod
#     def read(cls, params_file):
#         pnames, pmatrix = read_params_file(params_file)
#         return cls(pnames, pmatrix)


# Driver with command-line arguments
# ==================================
class Job(object):
    """Generic job class

    Attributes
    ----------
    model : Model instance with default parameters
    pnames : list of parameter names
    pmatrix : list of parameter sets (as a list of list)
    pdefault : default param values
    params : subset of model.params that are to be modified (only container change, but same individual Param ==> easy to modify)
    args : all arguments as returned by ArgumentParser.parse_args


    Methods
    -------
    __init__ : parse everything and initialize the above attributes
    run: basically the main() function
    """

    def __init__(self, model_parser=None, model_class=None, outdir_default="output", inidir=None, description=None, epilog=None, formatter_class=RawDescriptionHelpFormatter, **kwargs):
        """
        model_parser : parser for --model-args command-line argument (or after --)
        model_class : callable (function or Model class) that take a string as input argument (if model_parser is None)
            or the Namespace object returned by argparse.ArgumentParser.parse_args (if model_parser is provided).
        outdir_default : default directory for model output
        inidir : directory from which to start the executable
            The default behaviour is to execute the model from the output directory.
        description, epilog, formatter_class, **kwargs : passed to `argparse.ArgumentParser`
        """

        parser = argparse.ArgumentParser(description=description, epilog=epilog, formatter_class=formatter_class, **kwargs)

        group = parser.add_argument_group("Model-Specific")
        group.add_argument('--model-args', default="", help="model-specific command-line args")
        group.add_argument('--model-help', action="store_true", help="help on --model-args")

        group = parser.add_argument_group("Modified parameters")
        e_group = group.add_mutually_exclusive_group()
        e_group.add_argument('-p', '--params', default=[], type=params_parser, nargs='*', metavar="MODULE:BLOCK&NAME=VALUE", 
                           help="Modified parameters. Full specification: [MODULE:][BLOCK&]NAME=VALUE[,VALUE...]")

        e_group.add_argument('--params-file', help="Input parameter file. Header line of parameter names (with string quotation marks for each name), then one line per parameter set. Names and values are separated by empty spaces. Example:\n 'a' 'b' 'c'\n 1 2 3 \n 1.2 2.1 2.9")

        # group.add_argument('-m', '--module', help="Default module for --params", choices=["sico","rembo","climber2"])
        group.add_argument('-m', '--module', help="Default module for --params")
        group.add_argument('-b', '--block', help="Default block for --params")

        group.add_argument('--include-default', action="store_true", help="Perform a default control run in addition to perturbed simulation? (if --params is provided)")

        # group.add_argument('--params-file-save', help="Write modified parameters to a file, for later use.")

        group = parser.add_argument_group("Output directory, miscellaneous")

        group.add_argument('-o','--out-dir', 
                           help="Specify the output directory where all model \
                           output and parameter files will be stored (default = %(default)s)", default=outdir_default)

        group.add_argument('-a','--auto-dir', action="store_true", help="subdirectory will be generated \
                            based on the parameter arguments automatically (always true in batch mode)")

        group.add_argument('-1','--single', action="store_true", help="flatten output directory structure when ensemble size is 1")

        group.add_argument('-i','--interactive', action="store_true", 
                           help="Interactive submission: this will ask for user confirmation at various steps before running the model.")

        group.add_argument('--dry-run', action="store_true", help="Setup directories, but do not submit.")

        group.add_argument('-D','--clean', action="store_true", 
                           help="clean output directory by deleting any pre-existing files")

        group = parser.add_argument_group("Job submission (queue)")

        group.add_argument('--background', action="store_true",
                            help="execute the job as a background process; default is to run the job in the terminal")

        group.add_argument('-s', '--submit', action="store_true",
                            help="send the job to the queue; default is to run the job in the terminal")

        group.add_argument('--job-class', default="medium", help="job class, values depend on the queuing system used. On slurm (PIK): `squeue`, on loadleveler (old PIK): `llclass`")

        group.add_argument('--system', default="slurm", choices = ["slurm", "qsub", "loadleveler"], 
                           help="queueing system name, if `--submit` is passed. TODO: detect automatically based on machine architecture.")

        group.add_argument('-w','--wall', default="24", dest="wtime",
                            help="Wall clock time, specify the wall clock limit in the \
                            submit script ( '-w 1' means the job will be killed \
                            after 1 hour) (default: %(default)s)")

        # parse arguments
        # ---------------
         
        # -- syntax instead of model args?
        if '--' in sys.argv:
            assert '--model-args' not in sys.argv, "if --model-args is defined, the '--' separator is invalid"
            i = sys.argv.index('--')
            sys.argv = sys.argv[:i] + ['--model-args'] + [" ".join(sys.argv[i+1:])]

        # now parse arguments
        args = parser.parse_args() # general

        # now back to '--' syntax for easy printing
        if '--model-args' in sys.argv:
            i = sys.argv.index('--model-args')
            sys.argv = sys.argv[:i] + ['--'] + sys.argv[i+1].split()

        if args.model_help:
            if model_parser is not None:
                model_parser.parse_args(['--help'])
            else:
                print "No model help provided, pass-on command line arguments via `--model-args=my_string`"
                sys.exit()

        # model arg parser
        if model_parser is not None:
            model_args = model_parser.parse_args(args.model_args.split())
        else:
            model_args = args.model_args  # just pass on

        # model init
        # ----------
        assert model_class is not None, "must provide model class !"

        # initialize model
        model = model_class(model_args)

        # Parse parameter sets to be combined
        # -----------------------------------
        pnames, pmatrix = self.get_params_matrix(args)

        # Lookup pnames vs model, and extract actual Param instances from model
        # ---------------------------------------------------------------------
        params = Params()
        for pname in pnames:
            # TODO: lookup_param should be a model method
            param = lookup_param(pname, model.params, default_module=args.module, default_block=args.block)
            params.append(param)

        # for the record
        self.model = model
        self.args = args
        self.pnames = pnames
        self.pmatrix = pmatrix
        self.pdefault = [p.value for p in params] # default values
        self.params = params # Parameter list extracted from Model
        self.ini_dir = inidir


    def get_params_matrix(self, args):
        """Parse parameters based solely on parameter's input format, without 
        cross-checking with Model instance at that point.

        Parameters
        ----------
        args : Namespace as returned by ArgumentParser.parse_args 

        Returns
        -------
        pnames : list of parameter names
        pmatrix : list of parameter sets
            Each parameter set is a list/tuple of values
        """
        if args.params_file:
            # User-input parameter sets: combination done outside this model
            pnames, pmatrix = read_params_file(args.params_file)

        else:
            # args.params is a list of (pname, pvalues),  not yet combined
            # e.g. [("a", [1, 2, 3]), ("b", [1.1, 2.2, 3.3])]
            # Use zip to separate names from values: ["a", "b"] and [[1, 2, 3], [1.1, 2.2, 3.3]]
            if len(args.params) > 0:
                pnames, factors = zip(*args.params)
            else:
                pnames, factors = [], []

            # Combine all parameters as a list of parameter sets
            #  [[a1, a2, a3], [b1, b2]]   ---->
            #            [[a1, b1],   # set 1 (across modules)
            #             [a2, b1],   # set 2
            #             [a3, b1],   # set 3
            #             [a1, b2],   # set 4
            #             [a2, b2],   # set 5
            #             [a3, b2]]   # set 6
            pmatrix = combiner(factors)
                
            assert len(pmatrix) > 0, repr(factors) # this is always true, contains mind. []

        return pnames, pmatrix


    def job_summary(self):
        """Print job summary
        """
        args = self.args

        # Write summary to screen
        print
        print "Job script"
        print "----------"
        if hasattr(self.model, 'executable') and self.model.executable: 
            print "executable:", self.model.executable

        # print_params_summary(model, params_def, pmatrix, args)
        if len(self.params) == 0:
            print "default parameters"
        else:
            # print "default parameters (to be modified):"
            # print " "+"\n ".join(str(p) for p in self.params)
            print "modified parameters"
            if args.params_file:
                print "from file: "+args.params_file
            else: 
                # for now only factorial design
                print "factorial design"
                print str_factors(args.params)

            print "ensemble"
            print str_pmatrix(self.pnames, self.pmatrix)

            # add default parameter version?
            print "include default version: ", args.include_default

        print "number of simulations:", len(self.pmatrix)
        print "output directory:",args.out_dir,"(",("already exists"+" - will be deleted"*args.clean) \
            if os.path.exists(args.out_dir) else "to be created",")"    
        print


    def run(self):

        args = self.args
        pnames = self.pnames
        pmatrix = self.pmatrix

        self.job_summary()

        if args.interactive: ask_user()

        if len(self.params) > 0  and args.include_default:
            pmatrix = self.pdefault*args.include_default + pmatrix # default = empty set

        if os.path.exists(args.out_dir) and args.clean:
            shutil.rmtree(args.out_dir)

        return run_ensemble(self.model, pnames, pmatrix, args.out_dir,  interactive=False, dry_run=args.dry_run,
                            autodir=args.auto_dir, submit=args.submit, wtime=args.wtime, job_class=args.job_class, background=args.background, inidir=self.ini_dir, single=args.single)


# run a model for an ensemble of parameters
# =========================================
def run_ensemble(model, pnames, pmatrix, outdir, interactive=False, dry_run=False, autodir=False, submit=False, background=False, inidir=None, single=False, **job_args):
    """setup output directory and run ensemble

    model : Model instance - like
        This requires `update_params` and `setup_outdir` methods
    pnames : corresponding list of parameter name (p)
    pmatrix : list of list of parameter values (n x p)
    outdir : output directory
    interactive : interactive submission (default to False)
    dry_run : do not setup output directory nor execute/submit job
    autodir : bool, optional
        automatic naming of folders based on param names=
        default: False, use run number as name
    submit : submit to queue instead of terminal ?
    background : run as background in the terminal
    inidir : directory from which to launch the executable
    single : flatten the output directory structure where there is only one ensemble member.
    **job_args : passed to submit_job
    """
    outdir = os.path.abspath(outdir) + '/'

    if len(pmatrix) == 0:
        raise ValueError("Empty batch !") # make sure this does not happen

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # write job command to file, for the record
    with open(os.path.join(outdir, 'job.command'), 'w') as f:
        f.write(" ".join(sys.argv)+'\n')

    # write ensemble parameter file
    write_params_file(os.path.join(outdir, 'job.params'), pnames, pmatrix)

    # match params and model
    params = [lookup_param(pname, model.params) for pname in pnames]

    joblist = [] # if run is True

    N = len(pmatrix)
    single = single and N == 1  # single run in a flatten output directory?

    for i, pset in enumerate(pmatrix):

        # update default parameters for each module
        for j, val in enumerate(pset):
            params[j].value = val

        model.update_params(params) # redundant...

        print 
        # print "({}/{}):".format(i+1, len(batch)), ", ".join(str(p) for p in params) if len(params) > 0 else "default"
        print "({}/{}):".format(i+1, N), "(default)" if len(params) == 0 else ""
        if len(params) > 0:
            print " "+"\n ".join(str(model.params[model.params.index(p)]) for p in params if p.name)
            # print " "+"\n ".join(str(p) for p in params)

        # setup the output directory
        if single:
            outfldr = outdir
        else:
            if autodir:
                subfldr = autofolder(params, "")
            else:
                subfldr = "{:>05}".format(i)
            outfldr = outdir + subfldr

            print "sub-folder:", subfldr

        # if interactive: 
        #     response = ask_user(skip=True)
        #     if response == 's': continue

        if dry_run:
            continue

        print "Start simulation (submit to queue ? {})...".format(submit)

        if not os.path.exists(outfldr):
            os.makedirs(outfldr)

        # format initial directory
        ini_dir = outfldr if inidir is None else inidir

        if submit:
            job_id = model.submit(outfldr, ini_dir=ini_dir, **job_args)

        else:
            job_id = model.run(outfldr, ini_dir=ini_dir, background=background)
    
        joblist.append(subfldr)

    # Write the job list to a file
    # (make the output folder relative to the output/ directory)
    try:
        joblist1  = "\n".join(joblist) + '\n'
        if os.path.isfile(outdir+"batch"):
            open(outdir+"batch","a").write("\n"+joblist1)
        else:
            open(outdir+"batch","w").write(joblist1)
        
        print "Output folder(s):\n"
        print joblist1
        print "\n"
        
    except:
        print "Unable to write batch list to " + outdir

    return joblist

