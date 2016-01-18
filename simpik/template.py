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
import os, sys, shutil
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

    def set_timing(self, year0=None, years=None):
        raise NotImplementedError()

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
    dry_run : do not execute/submit job
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

        print 
        print "({}/{}):".format(i+1, len(batch)), ", ".join(str(p) for p in params) if len(params) > 0 else "default"

        # update default parameters for each module
        prog.update_params(params)

        # check that update went well
        print "Check:", ", ".join(str(prog.params[prog.params.index(p)]) for p in params if p.name) if len(params) > 0 else "default"

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
        exe, cmd_args = prog.setup_outdir(outfldr)

        if dry_run:
            continue
        print "Start simulation (submit to queue = {})...".format(submit)

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

def parse_param(string):
    """modified parameters as [MODULE:][BLOCK&]NAME=VALUE[,VALUE,...]

    where BLOCK and MODULE, if not provided, will be guessed from program's default parameters.
    To each MODULE corresponds a set of default parameters (and a parameter file).
    Note BLOCK is namelist-specific, and may be deprecated in future versions.
    """
    name, value = string.split('=')
    if ',' in value:
        values = value.split(',')
    else:
        values = [value]

    if ':' in name:
        module, name = name.split(':')
    else:
        module = None
    if '&' in name:
        block, name = name.split("&")
    else:
        block = None
    params = [Param(name=name, value=_parse_val(value), group=block, module=module) for value in values]
    return params

def parse_param_check(string):
    try:
        params = parse_param(string)
    except Exception as error:
        print "ERROR:",error.message
        raise
    return params


# run the model
class JobTemplate(object):
    """Generic job class

    Attributes
    ----------
    parser: ArgumentParser instance

    Methods
    -------
    add_prog_arguments: initialize program instance based on parsed parameters (to be subclassed)
    parse_prog: initialize program instance based on parsed parameters (to be subclassed)
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
        group.add_argument('-p', '--params', default=[], type=parse_param_check, nargs='*', metavar="MODULE:BLOCK&NAME=VALUE", 
                           help="Modified parameters. Full specification: [MODULE:][BLOCK&]NAME=VALUE[,VALUE...]")
        # group.add_argument('-m', '--module', help="Default module for --params", choices=["sico","rembo","climber2"])
        group.add_argument('-m', '--module', help="Default module for --params")
        group.add_argument('-b', '--block', help="Default block for --params")

        group.add_argument('--include-default', action="store_true", help="Perform a default control run in addition to perturbed simulation? (if --params is provided)")

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

    def parse_prog(self, args):
        """initialize program instance based on parsed parameters

        return : Program instance
            - `params` attribute (a list of all parameters)
            - `executable` attribute
            - `update_params` method
            - `setup_outdir` method
        """
        raise NotImplementedError("need to be subclassed")

    def parse_args_and_run(self):
        args = self.parser.parse_args()

        prog = self.parse_prog(args)

        # Get the runs as list of parameters not yet combined
        # [[a1, a2, a3], [b1, b2]] for parameters a and b, with 3 and 2 possible values, respectively
        # factors = program_parser.parse_params(args)
        factors = args.params

        for p in chain(*factors):
            if p.module is None: p.module = args.module
            if p.group is None: p.group = args.block

        # check against program parameters
        for plist in factors:
            p = plist[0]
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
            # set module attribute
            if p.module == None:
                for p in plist:
                    p.module = matches[0].module
            if p.group == None:
                for p in plist:
                    p.group = matches[0].group
        
        # print "Factors", factors

        # Combine all parameters as a list of parameter sets
        #            [[a1, b1],   # set 1 (across modules)
        #             [a2, b1],   # set 2
        #             [a3, b1],   # set 3
        #             [a1, b2],   # set 4
        #             [a2, b2],   # set 5
        #             [a3, b2]]   # set 6
        batch = combiner(factors)
        assert len(batch) > 0, repr(factors) # this is always true, contains mind. []

        print
        print "Job script"
        print "----------"
        print "executable:", prog.executable
        if len(batch) == 1 and len(batch[0]) == 0:
            print "parameters: default"
        else:
            print "modified parameters:"
            # for mod, params in groupby(chain(*factors), lambda p: p.module):
                # print "  {}".format(mod)
            for key, pars in groupby(chain(*factors), lambda p: p.key):
                print "  {} = {}".format(key, ", ".join([repr(p.value) for p in pars]))
            print "include default version: ", args.include_default
            batch = [[]]*args.include_default + batch # default = empty set
        print "number of simulations:", len(batch)
        print "output directory:",args.out_dir,"(",("already exists"+" - will be deleted"*args.clean) \
            if os.path.exists(args.out_dir) else "to be created",")"    
        print

        if args.interactive: ask_user()

        if os.path.exists(args.out_dir) and args.clean:
            shutil.rmtree(args.out_dir)

        return run_ensemble(prog, batch, args.out_dir,  interactive=args.interactive, dry_run=args.dry_run,
                     autodir=args.auto_dir, submit=args.submit, wtime=args.wtime, job_class=args.job_class, background=args.background)
