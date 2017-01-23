"""Job configuration: user interface
"""
#import simtools.model.params as mp
import argparse
import copy
import logging
import json
from simtools.model import Model, Param
from simtools.prior import Prior, GenericParam
#from simtools.job.prior import prior_parser
from simtools.parsetools import add_optional_argument, grepdoc
from simtools.job.filetype import (filetype_parser, getfiletype, filetype_as_dict)

def json_default(item):
    """`default` of json.dumps: returns serializable version of object
    """
    return json.loads(item.tojson())

class SubConfig(object):

    @classmethod
    def doc(cls, name):
        " return doc about one param "
        if not cls.__doc__:
            return ""
        return grepdoc(cls.__doc__, name)

    def add_argument(self, parser, name, **kwargs):
        """add argument to a parser instance, based on diagnosed default
        """
        return add_optional_argument(parser, name, 
                                     default=getattr(self, name), 
                                     help=self.doc(name), **kwargs)

    def parser_group(self, **kwargs):
        " generate parser for this param "
        parser = argparse.ArgumentParser(add_help=False, **kwargs)
        grp = parser.add_argument_group(self.__doc__.splitlines()[0])
        return parser, grp

    @property
    def parser(self):
        """feeling lucky, add all arguments...
        """
        parser, grp = self.parser_group()
        for p in self.__dict__:
            self.add_argument(grp, p)
        return parser

    def tojson(self, diff=False, **kwargs):
        if diff:
            cfg = self.diff()
        else:
            cfg = self.__dict__
        return json.dumps(cfg, **kwargs)

    def __iter__(self):
        return iter(self.__dict__)

    def diff(self):
        " dict of differences compared to default "
        cfg_def = type(self)()  # default config
        cfg = {k:getattr(self,k)
               for k in self if getattr(self, k) != getattr(cfg_def, k)}
        return cfg

    @classmethod
    def read(cls, file, root=None):
        cfg = json.load(open(file))
        cfg = cfg.pop(root, cfg) 
        self = cls()
        for k in self:
            if k in cfg:
                setattr(self, k, cfg[k])
        return self


    def update(self, cfg):
        """like update_known but raise an error if unknown args
        """
        unknown = self.update_known(cfg)
        if unknown:
            raise ValueError('{} :: unknown parameters: {}'.format( 
                self.__class__.__name__, 
                ", ".join(unknown.keys())))

    def update_known(self, cfg):
        """pick whatever is known in cfg to update and return the rest
        """
        cfg = cfg.copy()
        for k in self:
            setattr(self, k, cfg.pop(k, getattr(self, k)))
        return cfg


    def _assert_internals(self):
        """check internal consistency

        * parser namespace match class namespace
        * instance and class methods are disjoint
        """
        namespace = self.parser.parse_args(['--executable', self.executable])
        not_in_self = set(namespace.__dict__).difference(set(self))
        assert not not_in_self, "unknown argument in parser: "+repr(not_in_self)
        not_in_parser = set(self).difference(set(namespace.__dict__))
        if not_in_parser:
            logging.warn(repr(type(cls))+":: argument not in parser: "+repr(not_in_parser))
        to_be_renamed = set(self).intersection(set(type(self).__dict__))
        assert not to_be_renamed, \
            "conflicting method and argument names:"+repr(to_be_renamed)

    #def update(self, argv=None, namespace=None):
    #    """update with command line and return unknown params
    #    """
    #    namespace, unknown = self.parser.parse_known_args(argv, namespace)
    #    self.__dict__.update(namespace)
    #    return unknown


class FileTypeConfig(SubConfig):
    """model params filetype

    * file_type : model params file type
    * line_sep : separator for 'linesep' and 'lineseprev' file types
    * line_template : line template for 'linetemplate' file type
    * template_file : template file for 'template' file type
    * file_module : module to import with custom file type
    """
    def __init__(self, file_type="json", line_sep=" ", line_template=None, 
                 template_file=None, file_module=None):
        self.file_type = file_type
        self.line_sep = line_sep
        self.line_template = line_template
        self.template_file = template_file
        self.file_module = file_module

    def getfiletype(self):
        try:
            return getfiletype(**self.__dict__)
        except:
            return None


class ModelConfig(FileTypeConfig):
    """model config

    * executable : model executable (e.g. runscript etc)
    * args : model arguments. 
    * params : model parameters
    * default_file : default param file, only needed for certain file types (e.g. namelist)
    * paramsio : mode for passing parameters to model ("file" or "arg")

    * file_name : parameters file name, relatively to {rundir} (do not write if left empty). Note this will define the '{filename}' format tag
    * arg_template : format for params as command-line args. Set to empty string for no parameter passing
    """
    def __init__(self, executable=None, args=None, model_params=None, default_file=None, paramsio="arg", 
                 arg_template="--{name} {value}", file_name="params.json", **kwargs):
        self.executable = executable
        self.args = args
        self.default_file = default_file
        self.model_params = model_params or []
        self.paramsio = paramsio
        self.arg_template = arg_template
        self.file_name = file_name
        FileTypeConfig.__init__(self, **kwargs)

    @property
    def filetype(self):
        return FileTypeConfig(**{k:getattr(self, k) for k in FileTypeConfig()})

    @property
    def parser(self):
        parser, grp = self.parser_group(parents=[self.filetype.parser], conflict_handler='resolve')  # for -p, params...
        grp.add_argument("-x","--executable", help=self.doc('executable'), required=True)
        self.add_argument(grp, 'args')
        self.add_argument(grp, 'default_file')
        grp.add_argument('--model-params', '-p',
                         type=Param.parse,
                         help=Param.parse.__doc__,
                         metavar="NAME=VALUE",
                         nargs='*',
                         default = [str(p) for p in self.model_params])

        grp = parser.add_argument_group('param passing job --> model')
        self.add_argument(grp, 'paramsio', choices=["arg", "file"])
        self.add_argument(grp, 'file_name')
        self.add_argument(grp, 'arg_template')

        return parser

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

        if self.params:
            model.update({p.name:p.value for p in self.params})
                
        return model


class SlurmConfig(SubConfig):
    """slurm workload configuration

    * qos : queue
    """
    def __init__(self, qos=None, job_name=None, account=None, walltime=None):
        self.qos = qos
        self.job_name = job_name
        self.account = account
        self.walltime = walltime


class PriorConfig(SubConfig):
    """model parameters
    """
    def __init__(self, prior_params=None):
        self.prior_params = prior_params or []

    @property
    def parser(self):
        parser, grp = self.parser_group()
        grp.add_argument('--prior-params', '-p',
                         type=GenericParam.parse,
                         help=GenericParam.parse.__doc__,
                         metavar="NAME=SPEC",
                         nargs='*',
                         default = [str(p) for p in self.prior_params])
        return parser

    def getprior(self):
        return Prior(self.prior_params)

    def tojson(self, default=json_default, **kwargs):
        return super(PriorConfig, self).tojson(default=json_default, **kwargs)


class ObsConfig(SubConfig):
    """observational constraints
    """
    def __init__(self, constraints=[]):
        self.constraints = constraints

    @property
    def parser(self):
        parser, grp = self.parser_group()
        grp.add_argument('--likelihood', '-l', dest='constraints',
                         type=GenericParam.parse,
                         help=GenericParam.parse.__doc__,
                         metavar="NAME=SPEC",
                         nargs='*',
                         default = [str(p) for p in self.constraints])
        return parser



class JobConfig(PriorConfig, ModelConfig, SlurmConfig, ObsConfig):
    """global job configuration namespace
    """
    # some hierarchy in the config file
    _roots = ['prior', 'model', 'slurm', 'obs']

    def __init__(self, **kwargs): #prior=None, model=None, slurm=None, obs=None):
        " flat initialization : all arguments co-exist"

        # loop over PriorConfig, ModelConfig, etc... (all 4 super classes)
        for cls in JobConfig.__bases__:
            opts = {k:kwargs.pop(k) for k in cls() if k in kwargs}
            cls.__init__(self, **opts)

        # nothing should be left over
        if kwargs:
            raise TypeError('invalid arguments: '+repr(kwargs))

    @property
    def prior(self):
        return PriorConfig(**{k:getattr(self, k) for k in PriorConfig()})

    @property
    def model(self):
        return ModelConfig(**{k:getattr(self, k) for k in ModelConfig()})

    @property
    def slurm(self):
        return SlurmConfig(**{k:getattr(self, k) for k in SlurmConfig()})

    @property
    def obs(self):
        return ObsConfig(**{k:getattr(self, k) for k in ObsConfig()})

    @classmethod
    def read(cls, file):
        self = JobConfig()
        for name in cls._roots:
            grp = getattr(self, name)
            setattr(self, grp.read(file, name))
        return self

    def tojson(self, diff=True, sort_keys=True, indent=2, **kwargs):
        cfg = {}
        for name in self._roots:
            grp = getattr(self, name) # prior, model, etc..
            cfg[name] = json.loads(grp.tojson(diff=diff))
        return json.dumps(cfg, sort_keys=sort_keys, indent=indent, **kwargs)

    @property
    def parser(self):
        parents = [getattr(self, name).parser for name in self._roots]
        return argparse.ArgumentParser(add_help=False, parents=parents)


# default configuration (can be updated e.g. by job)
globalconfig = JobConfig()

# make additional checks w.r.t to parser and more
globalconfig._assert_internals()
            


# Programs
# --------
class Program(SubConfig):
    """A program is just a callable SubConfig instance relies on the global configuration.
    """
    def main(self):
        raise NotImplementedError()

    def __call__(self, argv=None):
        "normally done by Job (-c config.json), but just for checking"
        namespace = self.parser.parse_args(argv)
        self.update(namespace.__dict__)
        return self.main()

class EditModelConfig(ModelConfig, Program):
    """Setup job configuration via command-line

    * full : show full configuration, not only model
    * diff : only show difference to default config
    """
    def __init__(self, diff=False, full=False, **kwargs):
        self.full = full
        self.diff = diff 
        ModelConfig.__init__(self, **kwargs)

    @property
    def model(self):
        return ModelConfig(**{k:getattr(self, k) for k in ModelConfig()})

    @property
    def parser(self):
        parser = argparse.ArgumentParser(add_help=True, 
                                         parents=[globalconfig.model.parser])
        self.add_argument(parser, 'full')
        self.add_argument(parser, 'diff')
        return parser

    def main(self):
        if self.full:
            cfg = globalconfig
            cfg.update_known(json.loads(self.model.tojson()))
        else:
            cfg = self.model
        print( cfg.tojson(diff = self.diff, indent=2, sort_keys=True) )


#class EditConfig(JobConfig, Program):
#    """Setup job configuration via command-line
#
#    * only : only show one component, as a check
#    * full : show full config (by default only the differences are shown)
#    """
#    def __init__(self, only=None, full=False, **kwargs):
#        self.only = only
#        self.full = full
#        JobConfig.__init__(self, **kwargs)
#
#    @property
#    def config(self):
#        return JobConfig(**{k:getattr(self, k) for k in JobConfig()})
#
#    @property
#    def parser(self):
#        parser = argparse.ArgumentParser(add_help=True, parents=[globalconfig.parser])
#        parser.add_argument('--only', '-o', default=self.only, 
#                            help=self.doc('only'), choices=[JobConfig._roots])
#        self.add_argument(parser, 'full')
#        return parser
#
#    def main(self):
#        if self.only:
#            grp = getattr(self, self.only)
#        else:
#            grp = self.config
#        print( grp.tojson(diff = not self.full, indent=2, sort_keys=True) )


# main function
#editconfig = EditConfig()
modelconfig = EditModelConfig()

if __name__ == '__main__':
    #editconfig()
    modelconfig()
