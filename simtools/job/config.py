"""Job configuration: user interface
"""
#import simtools.model.params as mp
import argparse
import inspect
import copy
import logging
import json
from simtools.model import Model, Param
from simtools.prior import Prior, GenericParam
from simtools.job.tools import add_optional_argument, grepdoc, SubConfig, Job
from simtools.job.filetype import getfiletype


# Parameter groups
# ================

class FileTypeConfig(SubConfig):
    """model params filetype

    * file_type : model params file type
    * line_sep : separator for 'linesep' and 'lineseprev' file types
    * line_template : line template for 'linetemplate' file type
    * template_file : template file for 'template' file type
    * file_addon : module to import with custom file type
    """
    _root = 'filetype'

    def __init__(self, file_type="json", line_sep=" ", line_template=None, 
                 template_file=None, file_addon=None):
        self.file_type = file_type
        self.line_sep = line_sep
        self.line_template = line_template
        self.template_file = template_file
        self.file_addon = file_addon

    def getfiletype(self):
        try:
            return getfiletype(**self.__dict__)
        except:
            return None


class Paramsio(SubConfig):
    """model params io: passing job --> model

    * paramsio : mode for passing parameters to model ("file" or "arg")
    * file_name : parameters file name, relatively to {rundir} (do not write if left empty). Note this will define the '{filename}' format tag
    * arg_template : format for params as command-line args. Set to empty string for no parameter passing
    """
    _root = 'paramsio'

    def __init__(self, paramsio="arg", arg_template="--{name} {value}", file_name="params.json"):
        self.paramsio = paramsio
        self.arg_template = arg_template
        self.file_name = file_name

    @property
    def parser(self):
        parser = self._parser()
        grp = parser.add_argument_group(self._doc())
        self._add_argument(grp, 'paramsio', choices=["arg", "file"])
        self._add_argument(grp, 'file_name')
        self._add_argument(grp, 'arg_template')
        return parser


class SimuConfig(SubConfig):
    """model run

    * executable : model executable (e.g. runscript etc)
    * args : model arguments. 
    * params : model parameters
    * default_file : default param file, only needed for certain file types (e.g. namelist)
    """
    _root = 'simu'

    def __init__(self, executable=None, args=None, model_params=None, default_file=None, **kwargs):
        self.executable = executable
        self.args = args
        self.default_file = default_file
        self.model_params = model_params or []

    @property
    def parser(self):
        parser = self._parser(conflict_handler='resolve') # for -p, params
        grp = parser.add_argument_group(self._doc())
        grp.add_argument("-x","--executable", help=self._doc('executable'), required=False)
        self._add_argument(grp, 'args')
        self._add_argument(grp, 'default_file')

        def round_trip(p):
            par = Param.parse(p) # OK?
            return str(par)
        
        grp.add_argument('--params', '-p',
                         type=round_trip,
                         help=Param.parse.__doc__,
                         metavar="NAME=VALUE",
                         nargs='*',
                         dest='model_params',
                         default = self.model_params)
        return parser


class ModelConfig(SimuConfig, Paramsio, FileTypeConfig):
    """all model config objects
    """
    _root = 'model'

    def __init__(self, **kwargs):
        ModelConfig._super_init(self, **kwargs)

    parser = SubConfig.parser

    def getmodel(self):
        """return model
        """
        filetype = self.getfiletype()
        if self.default_file:
            params = filetype.load(open(self.default_file))
        else:
            params = []

        # no "paramsio" argument in Model class
        if self.paramsio == "file":
            arg_template = None
            file_name = self.file_name
        else:
            arg_template = self.arg_template
            file_name = None

        model = Model(self.executable, self.args, params, arg_template, file_name, filetype)

        if self.model_params:
            params = [Param.parse(p) for p in self.model_params]
            model.update({p.name:p.value for p in params})
                
        return model

#_add_bases_as_prop(ModelConfig)


class SlurmConfig(SubConfig):
    """slurm workload configuration

    * qos : queue
    """
    _root = 'slurm'
    def __init__(self, qos=None, job_name=None, account=None, walltime=None):
        self.qos = qos
        self.job_name = job_name
        self.account = account
        self.walltime = walltime


class PriorConfig(SubConfig):
    """prior distribution of model parameters
    """
    _root = 'prior'

    def __init__(self, prior_params=None):
        self.prior_params = prior_params or []

    @property
    def parser(self):
        parser = self._parser(conflict_handler='resolve') # -p
        grp = parser.add_argument_group(self._doc())
        grp.add_argument('--prior-params', '-p',
                         #type=lambda x: str(GenericParam.parse(x)), # parse back and forth
                         #type=GenericParam.parse,
                         help=GenericParam.parse.__doc__,
                         metavar="NAME=SPEC",
                         nargs='*',
                         default = [p for p in self.prior_params])
        return parser

    def getprior(self):
        return Prior([GenericParam.parse(p) for p in self.prior_params])


class ObsConfig(SubConfig):
    """observational constraints
    """
    _root = 'obs'
    def __init__(self, constraints=[]):
        self.constraints = constraints

    @property
    def parser(self):
        parser = self._parser()
        grp = parser.add_argument_group(self._doc())

        def round_trip(p):
            par = GenericParam.parse(p) # OK?
            return str(par)

        grp.add_argument('--likelihood', '-l', dest='constraints',
                         type=round_trip,
                         help=GenericParam.parse.__doc__,
                         metavar="NAME=SPEC",
                         nargs='*',
                         default = [p for p in self.constraints])
        return parser



class JobConfig(PriorConfig, ModelConfig, SlurmConfig, ObsConfig):
    """global job configuration namespace
    """
    # some hierarchy in the config file
    _roots = ['prior', 'model', 'slurm', 'obs']

    def __init__(self, **kwargs): #prior=None, model=None, slurm=None, obs=None):
        " flat initialization : all arguments co-exist"
        JobConfig._super_init(self, **kwargs)

    parser = SubConfig.parser


# make additional checks w.r.t to parser and more
globalconfig = JobConfig()
globalconfig._assert_internals()
del globalconfig  # just for debugging, remove !
            


# Programs
# --------
class Program(SubConfig):
    """Default configuration file

    * config_file : default configuration file
    """
    _root = 'config'

    def __init__(self, config_file=None):
        self.config_file = config_file

    @property
    def parser(self):
        parser = self._parser()
        self._add_argument(parser, 'config_file', aliases=('-c',))
        return parser

    def main(self):
        raise NotImplementedError()

    def __call__(self, argv=None):
        """Actual program execution
        """
        namespace = self.parser.parse_args(argv)
        # update global config
        if namespace.config_file:
            jobconfig = JobConfig.read(namespace.config_file)
            self._update_known(jobconfig.__dict__)
            namespace = self.parser.parse_args(argv) # parse again !
        self._update(namespace.__dict__)
        return self.main()


class EditConfig(JobConfig, Program):
    """Setup job configuration via command-line

    * only : pick a sub configuration to show
    * diff : only show difference to default config
    * flat : flatten config
    """
    _root = None
    _choices = [(cls._root, cls) for cls in JobConfig._units_mro()] + [('model', JobConfig), (None, JobConfig)]

    def __init__(self, only=None, diff=False, flat=False, **kwargs):
        self.only = only
        self.diff = diff 
        self.flat = flat 
        self._super_init(self, **kwargs)

    @property
    def parser(self):
        " only have parser for ModelConfig "
        parser = self._parser(add_help=True)
        self._add_argument(parser, 'flat')
        self._add_argument(parser, 'diff')
        self._add_argument(parser, 'only', nargs='*', choices=[x[0] for x in self._choices])
        return parser

    def main(self):
        cfg = {}
        for key in (self.only or [None]):
            grp = self._group(dict(self._choices)[key])
            cfg.update(json.loads(grp.tojson(diff=self.diff, flat=self.flat)))
        print( json.dumps(cfg, indent=2, sort_keys=True))

editconfig = EditConfig()

if __name__ == '__main__':
    editconfig()
