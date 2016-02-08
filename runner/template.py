"""Infrastructure to submit job to the cluster, run program in the background, 
and to provide command-line interface with the user. Program and JobTemplate should
be subclasses.

Design philosophy
-----------------
- each executable should have a corresponding Program class (see this class' help 
  for the API methods and attributes.). Note that only `executable`, `params`, 
  `update_params`, `setup_outdir` need to be implemented for the job script to work. 
  The rest (e.g. modules) is considered an implementation detail to ease re-use 
  of code, but is not required. 

- a Program class **may** use one or several module classes, that are related
  to an ensemble of functionalities several programs may share. It is much
  like a library. It is assumed that to each module is attached one parameter file.

- the default behaviour of a program subclass is to loop over its modules and call 
  their own `update_params` and `setup_outdir`, with an additional `exchange_params`
  method to handle the coupling between parameters. `exchange_params` is **not**
  part of the `API`: it is called internally by `update_params` in the parent
  Program class (and may be ignored in specific subclasses, up to the developers 
  taste).
"""
from __future__ import absolute_import
import os, sys, shutil, copy
import difflib
import argparse
from argparse import RawDescriptionHelpFormatter
from itertools import groupby, chain

from .tools import run_foreground, run_background, submit_job, ask_user
from .sampling import combiner
from .parameters import Param, Params


# Template for Program and Module classes
# ---------------------------------------
class Module(object):
    """Generic module class: this is an helper class, not part of the API
    """
    def __init__(self, params_file):
        " initialize a module from a parameter file "
        self.params = Params(params_file) 

    def update_params(self, params):
        """Update module parameters, and possibly resolve internal conflict
        """
        self.params.update(params) # udpate params
        # HERE: resolve conflicts?

    def setup_outdir(self, outdir):
        """setup output directory
        """
        pass

    def introspect(self):
        """Check internal consistency among parameters, raises an error if inconsistency detected.
        NOTE: it is up to the `update_params` method to actually modify their value.
        """
        pass


class Program(object):
    """Class to wrap an executable.

    Attributes
    ----------
    executable: str
        executable name
    cmd_args: str (empty by default)
        command-line arguments. May contain the space holder {outdir}
        to be formatted appropriately by `setup_outdir` method.
    params: `Params` instance 
        by default concatenate all modules' params (empty by default)
    modules: dict of Module instances (empty by default)
    """
    executable = None  
    modules = {}
    cmd_args = ""


    @property
    def params(self):
        """All program parameters"""
        return Params(chain(*[self.modules[mod_name].params for mod_name in self.modules]))


    def update_params(self, params):
        """update individual module parameters from command-line
        """
        for mod_name, mod_params in groupby(params, lambda p: p.module):
            self.modules[mod_name].update_params(mod_params)

        self.exchange_params()
        self.introspect()


    def exchange_params(self):
        """Internal parameter exchange among modules
        """
        pass


    def setup_outdir(self, outdir):
        """Setup program to run in the provided output directory

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

        for name, module in self.modules.iteritems():
            module.setup_outdir(outdir)

        return os.path.abspath(self.executable), self.cmd_args.format(outdir=outdir)


    def introspect(self):
        """Mark parameters with the proper `module` attribute, and check 
        for inconsistency among parameters.

        Raises an error if inconsistency detected.
        NOTE: it is up to the `update_params` method to actually modify their value.
        """
        # set module attribute for each param
        for m in self.modules:
            for p in self.modules[m].params:
                p.module = m

        for name in self.modules:
            self.modules[name].introspect()


    # The two methods below are not absolutely necessary, but useful to playaround 
    # ----------------------------------------------------------------------------
    def run(self, out_dir, background=False, cmd_args=None):
        """Run program locally (terminal or background)
        """
        exe, cmd = self.setup_outdir(out_dir)
        cmd_args = cmd + " " + (cmd_args or "")

        if background:
            job_id = run_background(exe, cmd_args, out_dir)
        else:
            job_id = run_foreground(exe, cmd_args)
        return job_id


    def submit(self, out_dir, cmd_args=None, **job_args):
        """Run program locally (terminal or background)
        """
        exe, cmd = self.setup_outdir(out_dir)
        cmd_args = cmd + " " + (cmd_args or "")
        return submit_job(exe, cmd_args, out_dir, **job_args)


# run a program for an ensemble of parameters
# -------------------------------------------
def run_ensemble(prog, batch, outdir, interactive=False, dry_run=False, autodir=False, submit=False, background=False, **job_args):
    """setup output directory and run ensemble

    prog : Program instance - like
        This requires `update_params` and `setup_outdir` methods defined as in `template.py`
    batch : list of parameter sets
        A paramter set is a Params instance (itself a list of Param instances)
    outdir : output directory
    autodir : bool, optional
        automatic naming of folders based on param names=
        default: False, use run number as name
    interactive : interactive submission (default to False)
    dry_run : do not setup output directory nor execute/submit job
    submit : submit to queue instead of background ?
    **job_args : passed to submit_job
    """
    outdir = os.path.abspath(outdir) + '/'

    if len(batch) == 0:
        raise ValueError("Empty batch !") # make sure this does not happen

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    joblist = [] # if run is True

    for i, params in enumerate(batch):


        # update default parameters for each module
        prog.update_params(params)

        print 
        # print "({}/{}):".format(i+1, len(batch)), ", ".join(str(p) for p in params) if len(params) > 0 else "default"
        print "({}/{}):".format(i+1, len(batch)), "(default)" if len(params) == 0 else ""
        if len(params) > 0:
            print " "+"\n ".join(str(prog.params[prog.params.index(p)]) for p in params if p.name)

        # setup the output directory
        if autodir:
            subfldr = autofolder(params, "")
        else:
            subfldr = "out.{:>05}".format(i)
        outfldr = outdir + subfldr

        print "sub-folder:", subfldr
        if interactive: 
            response = ask_user(skip=True)
            if response == 's': continue

        if dry_run:
            continue

        exe, cmd_args = prog.setup_outdir(outfldr)

        print "Start simulation (submit to queue ? {})...".format(submit)

        if submit:
            job_id = prog.submit(outfldr, **job_args)

        if background:
            job_id = run_background(exe, cmd_args, outfldr)
        elif submit:
            job_id = submit_job(exe, cmd_args, outfldr, **job_args)
        else:
            job_id = run_foreground(exe, cmd_args)
    
        joblist.append(subfldr)

    # Write the job list to a file
    # (make the output folder relative to the output/ directory)
    try:
        joblist1  = "\n".join(joblist)
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

# parse parameters from command-line
# ----------------------------------
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

def parse_param_name(string):
    """input_string : [MODULE:][BLOCK&]NAME
    parse and return name, block, module ; None if not provided

    where BLOCK and MODULE, if not provided, will be guessed from program's default parameters.
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

def parse_param_factors(string):
    """modified parameters as NAME=VALUE[,VALUE,...]
    where NAME will be passed as input to parse_param_name

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


def read_params_file(params_file, dtype=None):
    """read parameters matrix (from previous ensemble)
    dtype: parameters' type (by default: None means guessed from file)
    """
    import numpy as np
    pmatrix = np.genfromtxt(params_file, names=True, dtype=dtype)
    pnames = pmatrix.dtype.names
    return pnames, pmatrix

def write_params_file(params_file, pnames, pmatrix):
    """write parameters matrix
    """
    txt = str_pmatrix(pnames, pmatrix, max_rows=len(pmatrix), include_index=False)
    with open(params_file, 'w') as f:
        f.write(txt)
    # import numpy as np
    # np.savetxt(params_file, pmatrix, header=" ".join(pnames), comments="")
        
def lookup_param(p, prog):
    """Lookup matching default parameter among all program parameters

    p : Param instance
    prog : Program instance (with `params` and `executable` fields)

    return : corresponding default Param instance in prog
    """
    matches = [par for par in prog.params if par==p]
    if not matches:
        print sys.argv[0], ":: error :: invalid parameter for",prog.executable,":",repr(p.key)
        suggestions = difflib.get_close_matches(p.name, [par.name for par in prog.params])
        print "Do you mean: ", ", ".join(suggestions), "?"
        sys.exit(-1)
    elif len(matches) > 1:
        print sys.argv[0], ":: error :: several matches for param", repr(p.key), "in", prog.executable
        print "Matches:",matches
        print "Please specify module or block as [MODULE:][BLOCK&]NAME=VALUE[,VALUE...] or via --module and --block parameters"
        sys.exit(-1)
    assert len(matches) == 1
    return matches[0]


# string representation of parameter ensembles
def str_pmatrix(pnames, pmatrix, max_rows=10, include_index=True):
    """Pretty printing for a random ensemble of parameters
    """
    # compute some statistics using pandas' pretty printing
    # try:
    # except:
    # return str_pmatrix_pandas(pnames, pmatrix, max_rows=max_rows)
    return str_pmatrix_python(pnames, pmatrix, max_rows=max_rows, include_index=include_index)

def str_pmatrix_python(pnames, pmatrix, max_rows=10, include_index=True):
    """pretty-print parameters matrix like in pandas, but using only basic python functions
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


def str_pmatrix_pandas(pnames, pmatrix, max_rows=10):
    """matrix pretty printing using pandas
    """
    import pandas as pd
    pd.options.display.max_rows = max_rows

    df = pd.DataFrame(pmatrix, columns=pnames)
    df_str = str(df)

    lines = []
    lines.append(df_str)

    # large number of parameters, need to compute statistics
    if df.shape[0] > max_rows:
        # quantiles = df.quantile(q=[0.25, 0.5, 0.75])
        stats = pd.concat((pd.DataFrame({'min':df.min()}).T, 
                           # quantiles, 
                           pd.DataFrame({'max':df.max()}).T))
        # stats.index.name = "stats"
        stats_str = str(stats)
        lines.append('------- stats -------')
        lines.append(stats_str)
        lines.append('---------------------')

    return "\n".join(lines)


def str_factors(factors):
    """
    factors is a list of (pname, pvalues),  not yet combined
    e.g. [("a", [1, 2, 3]), ("b", [1.1, 2.2, 3.3])]
    """
    lines = ["factorial design"]
    for nm, values in factors:
        lines.append(" {} = {}".format(nm, ", ".join([repr(v) for v in values])))
    return "\n".join(lines)


# run the model
class JobTemplate(object):
    """Generic job class

    Attributes
    ----------
    parser: ArgumentParser instance

    Methods
    -------
    add_prog_arguments: initialize program instance based on parsed parameters (to be subclassed)
    init_prog: initialize program instance based on parsed parameters (to be subclassed)
    parse_args_and_run: basically the main() function.
    """

    def __init__(self, outdir_default="output", description=None):

        parser = argparse.ArgumentParser(description=description, formatter_class=RawDescriptionHelpFormatter)

        # Program specific
        # ================
        self.add_prog_arguments(parser)  # to be subclassed

        # Generic arguments
        # =================
        group = parser.add_argument_group("Modified parameters")
        e_group = group.add_mutually_exclusive_group()
        e_group.add_argument('-p', '--params', default=[], type=params_parser, nargs='*', metavar="MODULE:BLOCK&NAME=VALUE", 
                           help="Modified parameters. Full specification: [MODULE:][BLOCK&]NAME=VALUE[,VALUE...]")
        e_group.add_argument('--params-file', help="Input parameter file. Header line of parameter names (with string quotation marks for each name), then one line per parameter set. Names and values are separated by empty spaces. Example:\n 'a' 'b' 'c'\n 1 2 3 \n 1.2 2.1 2.9")

        # group.add_argument('-m', '--module', help="Default module for --params", choices=["sico","rembo","climber2"])
        group.add_argument('-m', '--module', help="Default module for --params")
        group.add_argument('-b', '--block', help="Default block for --params")

        group.add_argument('--include-default', action="store_true", help="Perform a default control run in addition to perturbed simulation? (if --params is provided)")

        group.add_argument('--params-file-save', help="Write modified parameters to a file, for later use.")

        group = parser.add_argument_group("Output directory, miscellaneous")

        group.add_argument('-o','--out-dir', 
                           help="Specify the output directory where all program \
                           output and parameter files will be stored (default = %(default)s)", default=outdir_default)

        group.add_argument('-a','--auto-dir', action="store_true", help="subdirectory will be generated \
                            based on the parameter arguments automatically (always true in batch mode)")

        group.add_argument('-i','--interactive', action="store_true", 
                           help="Interactive submission: this will ask for user confirmation at various steps before executing the program.")

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

        self.parser = parser


    def add_prog_arguments(self, parser):
        pass


    def init_prog(self, args):
        """initialize program instance based on parsed parameters

        return : Program instance
            - `params` attribute (a list of all parameters)
            - `executable` attribute
            - `update_params` method
            - `setup_outdir` method
        """
        raise NotImplementedError("need to be subclassed")


    def get_params_matrix(self, args):
        """Parse parameters based solely on parameter's input format, without 
        cross-checking with Program instance at that point.

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
            # User-input parameter sets: combination done outside this program
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

            print pmatrix

        return pnames, pmatrix


    def parse_job(self):
        """Parse job

        Returns
        -------
        prog : Program instance with default parameters
        params_def : default Params list (redundant...)
        batch : list of parameter sets
            A paramter set is a Params instance (itself a list of Param instances)
        args : all arguments as returned by ArgumentParser.parse_args
        """
        # Raw argument parser
        args = self.parser.parse_args()

        # Initialize default program version
        prog = self.init_prog(args)

        # Parse parameter sets to be combined
        pnames, pmatrix = self.get_params_matrix(args)

        # save to file if required
        if args.params_file_save:
            write_params_file(args.params_file_save, pnames, pmatrix)

        # Lookup corresponding default parameters, as a combination
        # between pnames and prog.params
        params_def = Params()
        for pname in pnames:
            name, block, module = parse_param_name(pname)
            # use default module is group is needed
            if module is None: module = args.module
            if block is None: block = args.block
            param_def = lookup_param(Param(name=name, group=block, module=module), prog)
            params_def.append(param_def)

        # Convert pmatrix to a list of list of parameter instances
        batch = []
        for i, parset in enumerate(pmatrix):
            # assert len(parset) == len(params_def), "Invalid params-file format. {} param nanes != {} param values in set {}".format(len(params_def), len(parset), i)
            params_set = Params()
            for j, param_def in enumerate(params_def):
                p = copy.copy(param_def)  # make sure default parameters are not modified
                p.value = pmatrix[i][j]
                params_set.append(p)
            batch.append(params_set)


        # Write summary to screen
        print
        print "Job script"
        print "----------"
        print "executable:", prog.executable

        # print_params_summary(prog, params_def, pmatrix, args)
        if len(params_def) == 0:
            print "default parameters"
        else:
            print "default parameters (to be modified):"
            print " "+"\n ".join(str(p) for p in params_def)
            if args.params_file:
                print "from file: "+args.params_file
                print str_pmatrix(pnames, pmatrix)
            else: 
                # for now only factorial design
                print str_factors(args.params)

        # add default parameter version?
        if len(params_def) > 0:
            print "include default version: ", args.include_default
            batch = [[]]*args.include_default + batch # default = empty set

        print "number of simulations:", len(batch)
        print "output directory:",args.out_dir,"(",("already exists"+" - will be deleted"*args.clean) \
            if os.path.exists(args.out_dir) else "to be created",")"    
        print

        return prog, params_def, batch, args


    def parse_args_and_run(self):

        prog, params_def, batch, args = self.parse_job()

        if args.interactive: ask_user()

        if os.path.exists(args.out_dir) and args.clean:
            shutil.rmtree(args.out_dir)

        return run_ensemble(prog, batch, args.out_dir,  interactive=args.interactive, dry_run=args.dry_run,
                     autodir=args.auto_dir, submit=args.submit, wtime=args.wtime, job_class=args.job_class, background=args.background)
