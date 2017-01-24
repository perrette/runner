"""File formats to pass parameters from job to model

* json : dict of {key : value} pairs written in json format
* linesep : one parameter per line, {name}{sep}{value} -- by default sep=" "
* lineseprev : one parameter per line, {value}{sep}{name} -- by default sep=" "
* linetemplate : one parameter per line, any format with {name} and {value} tag 
* template : based on template file, with {NAME} tags, one for each parameter

Note the "linetemplate" and "template" file types are WRITE-ONLY.
Additionally, there is an add-on system where any use-defined file format can be 
defined. Here an example with one of the existing file types:

    Two lines in a `rembo.py` module:

        from simtools.job.filetype import register_filetype, LineSeparator
        register_filetype("rembo", LineTemplate("{name:>10}:{value:24}"))

    For more complex formats you may want to define your own class. 
    It takes subclassing `ParamsFile.dumps`, and if needed `ParamsFile.loads`.
    Take a look at `simtools.model.params.LineTemplate` to learn how to proceed.
"""
import argparse
from simtools.model.params import JsonDict
from simtools.model.generic import LineSeparator, LineTemplate, TemplateFile
#from simtools.parsetools import CustomParser, grepdoc
#from simtools.job.addons import filetypes, protected_file_types


# FileType registration system 
# ============================
filetypes = {}
#choices = protected_file_types + filetypes.keys()
choices = ['json', 'linesep', 'lineseprev', 'linetemplate', 'template']

def register_filetype(name, filetype):
    if name in choices:
        raise ValueError("filetype name is protected: "+repr(name))
    if name in filetypes:
        raise ValueError("filetype name already exists: "+repr(name))
    if not hasattr(filetype, 'dumps'):
        raise TypeError("file type must have a `dumps` method")
    filetypes[name] = filetype
    choices.append(name)


def print_filetypes():
    print("Available filetypes:", ", ".join([repr(k) for k in choices+filetypes.keys()]))

# Params' file type
# -----------------
def getfiletype(file_type=None, line_sep=" ", line_template=None, template_file=None, file_addon=None):
    """Initialize file type

    * file_type : model params file type
    * line_sep : separator for 'linesep' and 'lineseprev' file types
    * line_template : line template for 'linetemplate' file type
    * template_file : template file for 'template' file type
    * file_addon : module to import with custom file type
    """
    if file_addon is not None:
        from importlib import import_module
        import_module(file_addon) # --> register_filetype command might be activated
            

    if file_type == "json":
        filetype = JsonDict()

    elif file_type == "linesep":
        filetype = LineSeparator(line_sep)

    elif file_type == "lineseprev":
        filetype = LineSeparator(line_sep, reverse=True)

    elif file_type == "linetemplate":
        if not line_template:
            raise ValueError("line_template is required for 'linetemplate' file type")
        filetype = LineTemplate(line_template)
    
    elif file_type == "template":
        if not template_file:
            raise ValueError("template_file is required for 'template' file type")
        filetype = TemplateFile(template_file)

    elif file_type in filetypes:
        filetype = filetypes[file_type]

    else:
        print_filetypes()
        raise ValueError("Unknown file type: "+str(file_type))
    return filetype
