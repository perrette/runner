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
from simtools.parsetools import CustomParser, grepdoc
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


#def get_filetype(name=None):
#    """Return filetype instance based on string
#    """
#    if name in filetypes:
#        return filetypes[name]
#
#    else:
#        raise ValueError("Unknown file type: "+repr(name))

def print_filetypes():
    print("Available filetypes:", ", ".join([repr(k) for k in choices+filetypes.keys()]))


# Params' file type
# -----------------
def getfiletype(file_type=None, line_sep=" ", line_template=None, template_file=None, file_module=None):
    """Initialize file type

    * file_type : model params file type
    * line_sep : separator for 'linesep' and 'lineseprev' file types
    * line_template : line template for 'linetemplate' file type
    * template_file : template file for 'template' file type
    * file_module : module to import with custom file type
    """
    if file_module is not None:
        from importlib import import_module
        import_module(file_module) # --> register_filetype command might be activated
            

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


# Build corresponding parser
doc = lambda param : grepdoc(getfiletype.__doc__, param)

filetype_parser = CustomParser(add_help=False, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
grp = filetype_parser.add_argument_group("model params filetype")
grp.add_argument("--file-type", default='json', 
                 help=doc('file_type')+"(default: %(default)s)")
grp.add_argument("--line-sep", default=" ", help=doc('line_sep'))
grp.add_argument("--line-template", help=doc('line_template'))
grp.add_argument("--template-file", help=doc('template_file'))
grp.add_argument("--file-module", help='import module with additional file type (addon)')

# add filetype class to parser namespace (inspect tells to look at function 
# to determine the key-word arguments)
filetype_parser.add_postprocessor(getfiletype, inspect=True, dest="filetype")


# filetype round-trip from the json format

def filetype_as_dict(filetype):
    """File-type --> json
    """
    if isinstance(filetype, JsonDict):
        cfg = {"file_type":"json"}

    elif isinstance(filetype, LineSeparator):
        if filetype.reverse:
            cfg = {"file_type":"lineseprev",
                   "line_sep" :filetype.sep}
        else:
            cfg = {"file_type":"linesep",
                   "line_sep" :filetype.sep}

    elif isinstance(filetype, LineTemplate):
        cfg = {"file_type":"linetemplate",
               "line_template" :filetype.line}
    
    elif isinstance(filetype, Template):
        cfg = {"file_type":"template",
               "template_file" :filetype.template_file}

    else:
        # maybe a custom file type?
        registered = [id(filetypes[k]) for k in filetypes]
        names = filetypes.keys()
        if id(filetype) in registered:
            cfg = {"file_type": names[registered.index(id(filetype))],
                   "file_module": filetype.__class__.__module__,
                   }
        else:
            raise ValueError("can't convert filetype to json !")

    return cfg

#def filetype_from_kw(**cfg):
#    return getfiletype(file_type=cfg.pop('file_type'),
#                       line_sep=cfg.pop('line_sep'," "), 
#                       line_template=cfg.pop('line_template',None), 
#                       template_file=cfg.pop('template_file', None), 
#                       file_module=cfg.pop('filetype_module', None))
