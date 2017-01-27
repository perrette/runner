"""File formats to pass parameters from job to model

* json : dict of {key : value} pairs written in json format
* linesep : one parameter per line, {name}{sep}{value} -- by default sep=" "
* lineseprev : one parameter per line, {value}{sep}{name} -- by default sep=" "
* linetemplate : one parameter per line, any format with {name} and {value} tag 
* template : based on template file, with {NAME} tags, one for each parameter

Note the "linetemplate" and "template" file types are WRITE-ONLY.
Additionally, there is an add-on system where any use-defined file format can be 
defined. Here an example how to use custom file types:

    A few lines in a `rembo.py` module:

        from simtools.register import register_filetype
        from simtools.model.generic import LineSeparator
        register_filetype("rembo", LineTemplate("{name:>10}:{value:24}"))
        from simtools.job import main
        main()

    For more complex formats you may want to define your own class. 
    It takes subclassing `ParamsFile.dumps`, and if needed `ParamsFile.loads`.
    Take a look at `simtools.model.params.LineTemplate` to learn how to proceed.
"""
import argparse
from simtools import register
from simtools.model import Model

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
grp.add_argument('--line_sep', help='separator for "linesep" and "lineseprev" file types')
grp.add_argument('--line_template', help='line template for "linetemplate" file type')
grp.add_argument('--template_file', help='template file for "template" file type')
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

model = argparse.ArgumentParser(add_help=False, parents=[filetype])

grp = model.add_argument_group('paramsio', 
                               description='param passing job --> model')
#grp.add_argument('--io-params', choices=["arg", "file"], default='arg',
#                 help='mode for passing parameters to model (default:%(default)s)')
grp.add_argument('--param-file',
                      help='model input parameter file, relatively to {rundir}. \
                 If provided, param passing via file instead of command arg.\
                 Note this might be used in model arguments as "{paramfile}"')
grp.add_argument('--param-arg', dest='arg_template', default='--{name} {value}',
                      help='format for params as command-line args')

grp = model.add_argument_group('model configuration')
grp.add_argument('--executable','-x', 
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


def getmodel(o):
    """return model
    """
    filetype = getfiletype(o)

    if not o.executable:
        model.error("argument --executable/-x is required")

    if o.default_file:
        params = filetype.load(open(o.default_file))
    else:
        params = []

    if o.file_name:
        o.arg_template = None  # only one or the other

    return Model(o.executable, o.args, params, o.arg_template, o.file_name, filetype)
