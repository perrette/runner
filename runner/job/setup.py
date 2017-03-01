"""Setup runner model interface

* interface.json : argparse argument values (loaded if --update is set)
* interface.pickle : initialized ModelInterface instance (loaded by `job run` etc)
"""

from __future__ import absolute_import, print_function
import argparse
import os, sys
import json, pickle, logging
import inspect
from importlib import import_module
import runner
import runner.model as mod
from runner.model import ModelInterface, Model
from runner.filetype import (LineSeparator, LineTemplate, TemplateFile, JsonFile)
from runner.ext.namelist import Namelist
from runner.job.tools import ObjectParser, Job



PREFIX = '' #'runner.'
#INTERFACE = PREFIX+'interface.pickle'
INTERFACE = None


# model file type
# ===============
filetxt="""File formats to pass parameters from job to model

* None : (default) dict of {key : value} pairs written in json format
* linesep : one parameter per line, {name}{sep}{value} -- by default sep=" "
* lineseprev : one parameter per line, {value}{sep}{name} -- by default sep=" "
* linetemplate : one parameter per line, any format with {name} and {value} tag 
* template : based on template file, with {NAME} tags, one for each parameter

Note the "linetemplate" and "template" file types are WRITE-ONLY.

Check out the formats already defined in runner.filetype and runner.ext
"""

choices = ['json', 'linesep', 'lineseprev', 'linetemplate', 'template', 'namelist']

filetype = argparse.ArgumentParser('[filetype]', add_help=False, 
                                   formatter_class=argparse.RawDescriptionHelpFormatter, description=filetxt)
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

# basic interface
# ---------------
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
grp.add_argument('--work-dir', default=None, 
                 help='where to execute the model from, by default current directory. Use "{}" for run directory.')

# custom model
# ------------
custommodel = argparse.ArgumentParser(add_help=False, parents=[])
grp = custommodel.add_argument_group('custom model')
grp.add_argument('-m','--interface', 
                 help='user-defined python module that contains custom model definition, or pickled model interface, or previous setup json file')


# run-specific configuration 
# --------------------------
modelconfig = argparse.ArgumentParser(add_help=False, parents=[])
grp = modelconfig.add_argument_group('run-specific model configuration')
grp.add_argument('--default-file', help='default param file, required for certain file types (e.g. namelist)')
grp.add_argument('--default-params', default=[], help='default param values (optional in most cases)')
grp.add_argument('command', nargs=argparse.REMAINDER, default=[], help='model executable and/or command-line arguments. Consumes all remaining arguments. \
`{}` and `{NAME}` will be replaced by \
    the run directory and corresponding parameter value, respectively.')


rawinterface_parser = argparse.ArgumentParser(add_help=False, 
                                       parents=[modelwrapper, custommodel, modelconfig])


def _getinterface(o):
    " model interface, except for command and defaults "

    # default model
    filetype = getfiletype(o, o.file_type, o.file_in)
    filetype_out = getfiletype(o, o.file_type_out, o.file_out)

    model = ModelInterface(
        work_dir=o.work_dir, 

        arg_out_prefix=o.arg_out_prefix, 
        arg_param_prefix=o.arg_prefix, 

        env_out=o.env_out, 
        env_prefix=o.env_prefix,

        filetype=filetype, 
        filename=o.file_in,
        filetype_output=filetype_out, 
        filename_output=o.file_out,
    )
    return model


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



def loadinterface(file=INTERFACE):
    " load interface after a setup "
    if not os.path.exists(file):
        parser.error("file "+file+" not found, run `job setup` first")
    _, ext = os.path.splitext(file)

    # load from json file (job setup's argparse)
    if ext == '.json':
        #logging.debug('load '+file)
        namespace = rawinterface.load(open(file))
        #logging.debug('get model from '+repr(namespace))
        model = rawinterface.get(namespace)

    # load from user-defined module?
    elif ext in ('.py','') or '::' in file:
        model = getcustominterface(file)

    # default: pickle returned by setup.py
    else:
        model = pickle.load(open(file, 'rb'))

    return model


def getdefaultparams(o, filetype=None, module=None):
    " default model parameters "
    if getattr(o, 'default_file', None):
        if filetype is None:
            rawinterface_parser.error('need to provide filetype along with default_file')
        default_params = filetype.load(open(o.default_file))
    else:
        default_params = getattr(o, 'default_params', [])
    return default_params


def updateinterface(model, o):
    " update command and defaults based on command line "
    if o.command and o.command[0] == '--':
        o.command = o.command[1:]
    model.args.extend(o.command)

    defaults = getdefaultparams(o, model.filetype)
    model.defaults.update(defaults)


def rawinterface_getter(o):
    """return model interface
    """
    # user-defined model?
    if o.interface:
        model = loadinterface(o.interface)

    else:
        model = _getinterface(o)

    updateinterface(model, o)

    return model

rawinterface = ObjectParser(rawinterface_parser, get=rawinterface_getter)

# Setup command
# =============


parser = argparse.ArgumentParser('setup', parents=[rawinterface_parser], description=__doc__)
parser.add_argument('--dir', default=os.path.curdir, help=argparse.SUPPRESS)
parser.add_argument('--to-pickle', help=argparse.SUPPRESS)
parser.add_argument('-f', '--force', action='store_true', 
                    help='overwrite any previous setup')
parser.add_argument('-u', '--update', action='store_true', help='update previous setup')


def pre(parser, argv=None):
    """set parser defaults
    """
    o, unknown = parser.parse_known_args(argv)
    if o.update:
        interfacejson = os.path.join(o.dir, PREFIX+'interface.json')
        import logging
        logging.info("set defaults: "+interfacejson)
        defaults = json.load(open(interfacejson))
        parser.set_defaults(force=True, **defaults)


def post(o):
    interfacepick = os.path.join(o.dir, 'interface.json')
    if os.path.exists(interfacepick) and not o.force:
        raise IOError("file already exists:"+interfacepick+" , use -f/--force or -u/--update")

    if getattr(o, "interface"):
        _, ext = os.path.splitext(o.interface)
        assert ext != '.json', '.json type interface can only be edited via -u/--update'

    # check it can be retrieved correctly
    mi = rawinterface.get(o)

    # ... and that it is pickable
    if o.to_pickle:
        pickle.dump(mi, open(o.to_pickle, 'wb'))
    else:
        pickle.dumps(mi) #, open(interfacepick, 'w'))

    # Save in json format for easy editing command-line args, for --update
    interfacejson = os.path.join(o.dir, "interface.json")
    rawinterface.dump(o, open(interfacejson, 'w'))

# wrap as JOB
main = Job(parser, post, pre)
main.register('setup', help='setup model interface')



# Object parser
# =============
# load interface following setup (not used in the setup command)
# --------------------------------------------------------------
interface_parser = argparse.ArgumentParser(add_help=False) #, parents=[modelconfig])
grp = interface_parser.add_argument_group('model interface')
grp.add_argument('-m', '--interface', 
                 default=INTERFACE, 
                 help='model interface. If not provided will look for interface[.py, .pickle, .json] (the last two are produced by job setup)')

def interface_getter(o):
    if not o.interface:
        exts = [".py", ".json", ".pickle"]
        for ext in exts:
            if os.path.exists("interface"+ext):
                o.interface = "interface"+ext
                break
        if not o.interface:
            interface_parser.error('-m/--interface: no interface found')
    return loadinterface(o.interface)


interface = ObjectParser(interface_parser, get=interface_getter)


if __name__ == '__main__':
    main()
