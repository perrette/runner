"""Job configuration: user interface
"""
#import simtools.model.params as mp
import json
from simtools.model import Model
from simtools.prior import Prior, GenericParam
#from simtools.job.prior import prior_parser
from simtools.parsetools import CustomParser
from simtools.job.filetype import (filetype_parser, getfiletype, filetype_as_dict)


# Model
# =====
def getmodel(executable, args=None, arg_template=None, file_name=None, 
             default_file=None, filetype=None):
    """get model config from command-arguments (first definition)
    """
    if default_file:
        params = filetype.load(open(default_file))
    else:
        params = []
    model = Model(executable, args, params, arg_template, filename, filetype)
    model._default_file = default_file  # dirty fix for now
    return model

def get_modelconfig(model):
    """return json-compatible model configuration --> first stage of round trip
    """
    cfg = filetype_as_dict(model.filetype)
    cfg.update({
        "executable": model.executable, 
        "args": model.args,
        "file_name": model.filename,
        "arg_template": model.arg_template,
    })
    if hasattr(model, '_default_file'):
        cfg["default_file"] = model._default_file
    return cfg

def read_modelconfig(config_file):
    """Read model from config file  --> close the round trip : ready for simu
    """
    dat = json.load(open(config_file))
    dat = dat.pop("model", dat) # remove any leading "model" key

    filetype = getfiletype(file_type=dat.pop('file_type'),
                           line_sep=dat.pop('line_sep'," "), 
                           line_template=dat.pop('line_template',None), 
                           template_file=dat.pop('template_file', None), 
                           file_module=dat.pop('filetype_module', None))

    return getmodel(dat.pop("executable"), 
                    dat.pop("args", None),
                    dat.pop("arg_template", None),
                    dat.pop("file_name", False),
                    dat.pop("default_file", None),
                    filetype)



# Params passing : Job --> Model
# ------------------------------
paramsio_parser = CustomParser(add_help=False, parents=[filetype_parser])
grp = paramsio_parser.add_argument_group("params passing job -> model")
grp.add_argument("--file-name", default=None,
                 help="parameters file name, relatively to {rundir} (do not write if left empty). Note this will define the '{filename}' format tag")
grp.add_argument("--arg-template", default="--{name} {value}",
                 help="format for params as command-line args. Set to empty string for no parameter passing (default: %(default)r)")

# Model configuration
# -------------------
model_parser = CustomParser(add_help=False, parents=[paramsio_parser])
grp = model_parser.add_argument_group("model config")

grp.add_argument("-x","--executable", help="model executable")

grp.add_argument("--args", default="",
                 help="model arguments. Alternnatively use the separator `--` and enter all arguments afterwards.")

grp.add_argument("--default-file", 
                 help="default param file, e.g. required by namelist format.If provided, default values will be picked from it. Must have same format as specified by --filetype")

model_parser.add_postprocessor(Model, dest='model',
                      args=["executable", "args", "filetype", "arg_template"], 
                      mapping={'filename':'file_name'})

# TODO: job batch : --> create a file with many `job run` commands, and one batch
# array file to execute them all.
# If no slurm, simply loop and define SLURM_ARRAY_TASK_ID environment variable for
# jobs in the background.


# Programs
# ========
def modelconfig(argv=None):
    "setup model configuration via command-line"
    parser = CustomParser(parents=[model_parser], 
                          description=modelconfig.__doc__)
    parser.add_argument("--config-file", 
                        help='configuration file to be updated')
    #parser.add_argument("--update","-u", 
    #                    help='update parameters already contained in the configuration file?')
    parser.add_argument("--indent", type=int, help='json output')
    #subs = parser.add_subparsers()
    #subs.add_parser('--', dest='model_args')

    import sys
    argv = argv or sys.argv[1:]

    if '--' in argv:
        i = argv.index('--')
        modelargs = argv[i+1:]
        argv = argv[:i]
    else:
        modelargs = []

    o = parser.parse_args(argv)
    o.args = o.args.split() + modelargs  # append modelargs

    o = parser.postprocess(o)  # filetype and model keys

    cfg_model = get_modelconfig(o.model)

    if o.config_file:
        cfg = json.load(open(o.config_file))
    else:
        cfg = {}
    cfg["model"] = cfg_model

    jsonstring = json.dumps(cfg, sort_keys=True, indent=o.indent)
    print(jsonstring)



prior_parser = CustomParser(add_help=False)
grp = prior_parser.add_argument_group('parameters')
x = grp.add_mutually_exclusive_group()
x.add_argument('--params-prior', '-p',
                         type=GenericParam.parse,
                         help=GenericParam.parse.__doc__,
                         metavar="NAME=SPEC",
                         nargs='*',
                         default = [])

prior_parser.add_postprocessor(Prior, dest='prior',
                               mapping={'params':'params_prior'})


# TODO: merge "params" and "prior" params ?
# ==> unified Param type with many fields OR just use current Param which is flexible
# and Param.toprior() to define prior distribution

#prior_parser.add_argument('--prior-file', dest="config_file", help=argparse.SUPPRESS)
##grp.add_argument('--config-file', '-c', dest="config_file", help="configuration file")

#prior_parser.add_postprocessor(getprior, inspect=True, dest='prior')


# Prior parameters
# ----------------
def priorconfig(argv=None):
    """create or update prior parameters
    """
    parser = CustomParser(description=priorconfig.__doc__, 
                          parents=[prior_parser])
    parser.add_argument("--config-file", 
                        help='configuration file to be updated')
    parser.add_argument("--indent", type=int)
    parser.add_argument('-a','--append', action='store_true',
                        help='append new params to config-file parameters instead of replacing')
    x = parser.add_mutually_exclusive_group()
    x.add_argument('-o','--only-params', nargs='*', 
                     help="filter out all but these parameters (with --append)")
    x.add_argument('-e', '--exclude-params', nargs='*', 
                     help="filter out these parameters (with --append)")

    o = parser.parse_args(argv)
    o = parser.postprocess(o)

    # append any new parameter to old config
    if o.config_file and o.append:
        # take what is good from old config file
        oldprior = Prior.read(config_file)
        if o.only_params:
            oldprior.filter_params(o.only_params, keep=True)
        if o.exclude_params:
            oldprior.filter_params(o.exclude_params, keep=False)

        # now see if there is anything to add to the new prior params
        names = [p.name for p in o.prior]
        for p in oldprior.params:
            if p.name not in names:
                o.prior.params.append(p)
                names.append(p.name)

    # config
    cfg_params = [json.loads(p.tojson()) for p in o.prior.params]

    # update old config file that may contain model config etc.
    if o.config_file:
        cfg = json.load(open(o.config_file))
    else:
        cfg = {}
    cfg["params"] = cfg_params
    jsonstring = json.dumps(cfg, sort_keys=True, indent=o.indent)
    print(jsonstring)
