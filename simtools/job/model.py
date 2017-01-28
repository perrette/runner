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
from simtools.model import Param, GenericModel, ARG_TEMPLATE, OUT_TEMPLATE, CustomModel
from simtools.filetype import JsonDict, LineSeparator, LineTemplate, TemplateFile

# model file type
# ===============
choices = ['json', 'linesep', 'lineseprev', 'linetemplate', 'template']
for c in choices:
    if c in register.filetypes:
        warnings.warn('registered file type overwrites a default: '+c)

filetype = argparse.ArgumentParser('[filetype]', add_help=False, 
                                   formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
grp = filetype.add_argument_group('filetype', description='file formats to pass parameters from job to model. Enter --help-file-type to see how to register custom filetypes')

grp.add_argument('--file-type', help='model params file type (including registered custom)', 
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

model_parser = argparse.ArgumentParser(add_help=False, parents=[filetype])

grp = model_parser.add_argument_group('interface', description='job to model communication')
#grp.add_argument('--io-params', choices=["arg", "file"], default='arg',
#                 help='mode for passing parameters to model (default:%(default)s)')
grp.add_argument('--param-file', 
                      help='model input parameter file, relatively to {rundir}. \
                 If provided, param passing via file instead of command arg.\
                 Note this might be used in model arguments as "{paramfile}"')
grp.add_argument('--param-arg', dest='arg_template', default=ARG_TEMPLATE,
                      help='format for params as command-line args')
grp.add_argument('--environ', action='store_true', 
                 help='pass parameters as environment variables')
grp.add_argument('--out-arg', dest='out_template', default=OUT_TEMPLATE,
                      help='format for params as command-line args')

grp = model_parser.add_argument_group('model configuration')
grp.add_argument('--executable','-x', default='echo',
                      help='model executable (e.g. runscript etc)')
grp.add_argument('--args', 
                 help='model arguments (quoted).\
can also be passed after "--" separator. \
allowed tags filled by job: \
    {expdir} (super folder), \
    {rundir} (ensemble member folder), \
    {runtag} (base name of {rundir}), \
    {paramfile} (parameter set for one model), \
    {runid} (simulation number in the ensemble). \
tags can also be formatted according to python rules, \
e.g. {runid:0>6} to prefix runid with zeroes, total 6 digits')

grp.add_argument('--default-file', help='default param file, required for certain file types (e.g. namelist)')


def getdefaultparams(o, filetype=None, module=None):
    " default model parameters "

    _default_params = getattr(module, '_default_params', None)

    if o.default_params:
        params = o.default_params

    elif o.default_file:
        if filetype is None:
            model_parser.error('need to provide filetype along with default_file')
        params = filetype.load(open(o.default_file))

    elif callable(_default_params):
        kw = _default_params()
        params = [Param(name, kw[name]) for name in sorted(kw.keys())]

    else:
        params = []

    return params


def getcustommodel(o):
    """get custom model as generated by job install
    """
    module = import_module(o.module) 

    make_command = getattr(module, 'make_command', None)
    setup = getattr(module, 'setup_output_dir', None)
    getvar = getattr(module, 'getvar', None)

    filetype = getfiletype(o)
    params = getdefaultparams(o, filetype, module)

    if setup is None:
        warnings.warn('no setup_output_dir found in module')

    return CustomModel(o.executable, make_command, setup, getvar, o.args, filetype=filetype, params=params)


def getgenericmodel(o):
    filetype = getfiletype(o)
    params = getdefaultparams(o, filetype)
    model = GenericModel(o.executable, o.args, params, o.arg_template, o.out_template, o.param_file, filetype=filetype, setenviron=o.environ)
    return model


def getmodel(o):
    """return model
    """
    # generic model
    if getattr(o,'module',None):
        model = getcustommodel(o)
    else:
        model = getgenericmodel(o)
    return model


#install = argparse.ArgumentParser(add_help=False, description=__doc__, parents=[model_parser])
#install.add_argument('-m','--module', default='model', help='new module to be created')
#install.add_argument('-f','--force', action='store_true', help='overwrite any existing module')
#install.add_argument('-p', '--default-params', nargs='+', metavar="NAME=VALUE", type=Param.parse)
#
#
#def install_post(o):
#    from simtools.templates.render import render_module
#
#    modulefile = o.module.replace('/','.') + '.py'
#    if os.path.exists(modulefile) and not o.force:
#        install.error("module file already exists:"+modulefile)
#
#    model = getgenericmodel(o)
#    modulesource = render_module(model)
#
#    try:
#        with open(modulefile, 'w') as f:
#            f.write(modulesource)
#    except:
#        print(modulesource)
#        raise
#    os.system('chmod +x '+modulefile)
#
#    print(modulefile,'created')
#
#register.register_job('install', install, install_post, 
#                      help='generate model script that works with job')
