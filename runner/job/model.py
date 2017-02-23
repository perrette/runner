"""File formats to pass parameters from job to model

* None : (default) dict of {key : value} pairs written in json format
* linesep : one parameter per line, {name}{sep}{value} -- by default sep=" "
* lineseprev : one parameter per line, {value}{sep}{name} -- by default sep=" "
* linetemplate : one parameter per line, any format with {name} and {value} tag 
* template : based on template file, with {NAME} tags, one for each parameter

Note the "linetemplate" and "template" file types are WRITE-ONLY.

Check out the formats already defined in runner.filetype and runner.ext
"""
from __future__ import absolute_import, print_function
import argparse
import os, sys
import json
import inspect
from importlib import import_module
import runner
import runner.model as mod
from runner.job import register
from runner.model import ModelInterface, Model
from runner.param import MultiParam, Param, DiscreteParam
from runner.filetype import (LineSeparator, LineTemplate, TemplateFile, JsonFile)
from runner.ext.namelist import Namelist


# model file type
# ===============
choices = ['json', 'linesep', 'lineseprev', 'linetemplate', 'template', 'namelist']

filetype = argparse.ArgumentParser('[filetype]', add_help=False, 
                                   formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
grp = filetype.add_argument_group('filetype', description='file formats to pass parameters from job to model. Enter --help-file-type for additional info')

grp.add_argument('--file-type', help='model params file type', choices=choices)
grp.add_argument('--file-type-out', help='model output file type',
                 choices=["json","linesep","lineseprev","namelist"])
grp.add_argument('--line-sep', help='separator for "linesep" and "lineseprev" file types')
grp.add_argument('--line-template', help='line template for "linetemplate" file type')
grp.add_argument('--template-file', help='template file for "template" file type')
grp.add_argument('--help-file-type', help='print help for filetype and exit', action='store_true')


def _print_filetypes():
    print("Available filetypes:", ", ".join([repr(k) for k in choices]))


def getfiletype(o, file_type=None, file_name=None):
    """Initialize file type
    """
    if o.help_file_type:
        filetype.print_help()
        filetype.exit(0)

    if file_type is None:
        file_type = o.file_type

    if not file_type:
        # first check vs extension
        _, file_type = os.path.splitext(file_name or o.file_in or "")

    if not file_type or file_type == '.txt':
        # default to basic '{name} {value}' on each line if no extension
        file_type = 'linesep'

    if file_type in ("json", ".json"):
        #ft = json
        ft = JsonFile()

    elif file_type == "linesep":
        ft = LineSeparator(o.line_sep)

    elif file_type == "lineseprev":
        ft = LineSeparator(o.line_sep, reverse=True)

    elif file_type == "linetemplate":
        if not o.line_template:
            raise ValueError("line_template is required for 'linetemplate' file type")
        ft = LineTemplate(o.line_template)
    
    elif file_type == "template":
        if not o.template_file:
            raise ValueError("template_file is required for 'template' file type")
        ft = TemplateFile(o.template_file)

    elif file_type in ("namelist", ".nml"):
        ft = Namelist()

    else:
        _print_filetypes()
        raise ValueError("Unknown file type or extension: "+str(file_type))
    return ft


# model runs
# ==========

modelwrapper = argparse.ArgumentParser(add_help=False, parents=[filetype])

grp = modelwrapper.add_argument_group('interface', description='job to model communication')
#grp.add_argument('--io-params', choices=["arg", "file"], default='arg',
#                 help='mode for passing parameters to model (default:%(default)s)')
grp.add_argument('--file-in','--file-name', 
                      help='param file name to pass to model, relatively to {rundir}. \
                 If provided, param passing via file instead of command arg.')
grp.add_argument('--file-out', 
                      help='model output file name, relatively to {rundir}. \
                 If provided, param passing via file instead of command arg.')
grp.add_argument('--arg-out-prefix', default=None,
                      help='prefix for output directory on the command-line. None by default.')
grp.add_argument('--arg-prefix', default=None,
                 help='prefix for passing param as command-line, e.g. `--{} ` where `{}` will be replaced by param name. None by default.')
grp.add_argument('--env-prefix', default=None,
                 help='prefix for environment variables')
grp.add_argument('--env-out', default=mod.ENV_OUT,
                 help='environment variable for output (after prefix) (default:%(default)s)')

custommodel = argparse.ArgumentParser(add_help=False, parents=[])
grp = custommodel.add_argument_group('user-customed model')
grp.add_argument('-m','--user-module', 
                 help='user-defined python module that contains custom model definition')

modelconfig = argparse.ArgumentParser(add_help=False, parents=[custommodel])
grp = modelconfig.add_argument_group('model configuration')
grp.add_argument('--default-file', help='default param file, required for certain file types (e.g. namelist)')
grp.add_argument('--work-dir', default=None, 
                 help='where to execute the model from, by default current directory. Use "{}" for run directory.')
modelconfig.add_argument('command', metavar='...', nargs=argparse.REMAINDER, default=[], help='model executable and its command-line arguments (need to be last on the command-line, possibly separated from other arguments with `--`). \
`{}` and `{NAME}` will be replaced by \
    the run directory and corresponding parameter value, respectively. \
    See also --arg-out-prefix, --arg-prefix')


model_parser = argparse.ArgumentParser(add_help=False, 
                                       parents=[modelwrapper, modelconfig])

def getdefaultparams(o, filetype=None, module=None):
    " default model parameters "
    if getattr(o, 'default_file', None):
        if filetype is None:
            model_parser.error('need to provide filetype along with default_file')
        default_params = filetype.load(open(o.default_file))
    else:
        default_params = []
    return default_params


def getcustominterface(user_module):
    if '::' in user_module:
        user_module, name = user_module.split('::')
    else:
        name = None

    if os.path.exists(user_module):
        sys.path.insert(0, os.path.dirname(user_module))
        user_module = os.path.basename(user_module)
    else:
        sys.path.insert(0, os.getcwd())

    user_module, ext = os.path.splitext(user_module)
    m = import_module(user_module)

    if name:
        return getattr(m, name)

    interfaces = inspect.getmembers(m, lambda x: isinstance(x, ModelInterface))
    if not interfaces:
        modelconfig.error('no runner.model.ModelInterface instance found')
    elif len(interfaces) > 1:
        logging.warn('more than one runner.model.ModelInterface instance found, pick one')
    return interfaces[0][1]


def getinterface(o):
    """return model interface
    """
    if o.command and o.command[0] == '--':
        o.command = o.command[1:]

    # user-defined model?
    if o.user_module:
        model = getcustominterface(o.user_module)
        # append any new arguments
        if o.command:
            model.args.extend(o.command)
        return model

    # default model
    modelargs = {}

    filetype = getfiletype(o, o.file_type, o.file_in)
    filetype_out = getfiletype(o, o.file_type_out, o.file_out)

    modelargs.update(dict(
        filetype=filetype, filename=o.file_in,
        filetype_output=filetype_out, filename_output=o.file_out,
    ))

    modelargs.update( dict(args=o.command, 
                           work_dir=o.work_dir, 
                           arg_out_prefix=o.arg_out_prefix, arg_param_prefix=o.arg_prefix, 
                           env_out=o.env_out, env_prefix=o.env_prefix) )

    model = ModelInterface(**modelargs)

    return model
