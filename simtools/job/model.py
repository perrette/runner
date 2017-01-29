"""Generate a customizable run script to work with your model.

File formats to pass parameters from job to model

* None : (default) dict of {key : value} pairs written in json format
* linesep : one parameter per line, {name}{sep}{value} -- by default sep=" "
* lineseprev : one parameter per line, {value}{sep}{name} -- by default sep=" "
* linetemplate : one parameter per line, any format with {name} and {value} tag 
* template : based on template file, with {NAME} tags, one for each parameter

Note the "linetemplate" and "template" file types are WRITE-ONLY.

Check out the formats already defined in simtools.filetype and simtools.ext
"""
from __future__ import absolute_import, print_function
import argparse
import os
import json
from importlib import import_module
import simtools
from simtools import register
import simtools.model as mod
from simtools.model import Param, Model, CustomModel, ParamsFile
from simtools.filetype import (JsonDict, LineSeparator, LineTemplate, 
                               TemplateFile, FileTypeWrapper)


# model file type
# ===============
choices = ['json', 'linesep', 'lineseprev', 'linetemplate', 'template']
for c in choices:
    if c in register.filetypes:
        warnings.warn('registered file type overwrites a default: '+c)

filetype = argparse.ArgumentParser('[filetype]', add_help=False, 
                                   formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
grp = filetype.add_argument_group('filetype', description='file formats to pass parameters from job to model. Enter --help-file-type to see how to register custom filetypes')

grp.add_argument('--file-type', help='model params file type (including registered custom)', default=mod.FILETYPE,
                 choices=choices+register.filetypes.keys())
grp.add_argument('--line-sep', help='separator for "linesep" and "lineseprev" file types')
grp.add_argument('--line-template', help='line template for "linetemplate" file type')
grp.add_argument('--template-file', help='template file for "template" file type')
grp.add_argument('--help-file-type', help='print help for filetype and exit', action='store_true')


def _print_filetypes():
    print("Available filetypes:", ", ".join([repr(k) for k in choices]))


def getfiletype(o):
    """Initialize file type
    """
    if o.help_file_type:
        filetype.print_help()
        filetype.exit(0)

    if o.file_type is None:
        return None

    if o.file_type in register.filetypes:
        ft = register.filetypes[o.file_type]

    elif o.file_type == "json":
        ft = JsonDict()

    elif o.file_type == "linesep":
        ft = LineSeparator(o.line_sep)

    elif o.file_type == "lineseprev":
        ft = LineSeparator(o.line_sep, reverse=True)

    elif o.file_type == "linetemplate":
        if not o.line_template:
            raise ValueError("line_template is required for 'linetemplate' file type")
        ft = LineTemplate(o.line_template)
    
    elif o.file_type == "template":
        if not o.template_file:
            raise ValueError("template_file is required for 'template' file type")
        ft = TemplateFile(o.template_file)

    else:
        _print_filetypes()
        raise ValueError("Unknown file type: "+str(o.file_type))
    return ft


# model runs
# ==========

modelcommand = argparse.ArgumentParser(add_help=False, parents=[filetype])

grp = modelcommand.add_argument_group('interface', description='job to model communication')
#grp.add_argument('--io-params', choices=["arg", "file"], default='arg',
#                 help='mode for passing parameters to model (default:%(default)s)')
grp.add_argument('--file-name', default=mod.FILENAME, 
                      help='file name to pass to model, relatively to {rundir}. \
                 If provided, param passing via file instead of command arg.\
                 Note this might be used in model arguments as "{paramfile}"')
grp.add_argument('--arg-out-prefix', default=mod.ARG_OUT_PREFIX,
                      help='format for params as command-line args (default=%(default)s)')
grp.add_argument('--arg-param-prefix', default=mod.ARG_PARAM_PREFIX,
                      help='prefix for passing param as command-line (default=%(default)s)')
grp.add_argument('--env-prefix', default=mod.ENV_PREFIX,
                 help='prefix for environment variables (default:%(default)s)')
grp.add_argument('--env-out', default=mod.ENV_OUT,
                 help='environment variable for output (after prefix) (default:%(default)s)')
grp.add_argument('--init-dir', default=None, 
                 help='where to execute the model from, by default current directory')

modelconfig = argparse.ArgumentParser(add_help=False, parents=[])
grp = modelconfig.add_argument_group('model configuration')
grp.add_argument('--executable','-x', default='echo',
                      help='model executable (e.g. runscript etc)')
grp.add_argument('--args', 
                 help='model arguments (quoted).\
allowed tags filled by job: \
    {expdir} (super folder), \
    {rundir} (ensemble member folder), \
    {runtag} (base name of {rundir}), \
    {paramfile} (parameter set for one model), \
    {runid} (simulation number in the ensemble). \
tags can also be formatted according to python rules, \
e.g. {runid:0>6} to prefix runid with zeroes, total 6 digits')

grp.add_argument('--default-file', help='default param file, required for certain file types (e.g. namelist)')

model_parser = argparse.ArgumentParser(add_help=False, 
                                       parents=[modelcommand, modelconfig])

#grp = model_parser.add_argument_group('user-defined module')
#grp.add_argument('--module-file', help='')


def getdefaultparams(o, filetype=None, module=None):
    " default model parameters "
    if o.default_file:
        if filetype is None:
            model_parser.error('need to provide filetype along with default_file')
        params = filetype.load(open(o.default_file))
    else:
        params = []
    return params


def getmodel(o):
    """return model
    """
    # check register first
    modelargs = register.model.copy() # command, setup, getvar, filetype

    loads = modelargs.pop('loads')
    dumps = modelargs.pop('dumps')
    if loads or dumps:
        filetype = FileTypeWrapper(dumps, loads)
    else:
        filetype = getfiletype(o)

    params = getdefaultparams(o, filetype)

    modelargs.update( dict(executable=o.executable, args=o.args, params=params, 
                 filetype=filetype, filename=o.file_name,
                 arg_out_prefix=o.arg_out_prefix, arg_param_prefix=o.arg_param_prefix, 
                 env_out=o.env_out, env_prefix=o.env_prefix,
                 init_dir=o.init_dir) )

    model = CustomModel(**modelargs)

    return model
