import json
import os, sys
from jinja2 import Template

TEMPLATEDIR = os.path.dirname(__file__)

def getmodeldict(model):
    """transform "model" into a dict
    """
    string = json.dumps(model.__dict__, 
                        #default=lambda x: x.__dict__ if hasattr(x, '__dict__') else None)
                        default=lambda x: json.loads(x.tojson()) if hasattr(x, 'tojson') else None)
    return json.loads(string)

def render_module(model):
    """Model instance
    """
    filename = os.path.join(TEMPLATEDIR, 'model.py.template')
    moduletemplate = open(filename).read()
    template = Template(moduletemplate)

    modeldict = getmodeldict(model)
    for p in modeldict["params"]:
        p["type"] = type(p["default"]).__name__
        p["default"] = repr(p["default"])  # escape strings

    if model.out_template:
        modeldict["out_template"] = model.out_template.format('{}', rundir='{}')
    else:
        modeldict["out_template"] = ""

    if model.arg_template:
        modeldict["arg_template"] = model.arg_template.format("{name}","{value}", 
                                                              name="{name}", 
                                                              value="{value}")
    else:
        modeldict["arg_template"] = ""

    js = {
        "model": modeldict,
        "command": " ".join(sys.argv)
    }

    # add file type info
    if model.filetype:
        ft = model.filetype
        cls = type(ft)

        # module
        if cls is type(os):
            js.update({
                "filetype_module": ft.__name__,
            })

        else:
            js.update({
                "filetype_module": cls.__module__,
                "filetype_class": getattr(ft, '__name__', cls.__name__),
                "filetype_kwargs": {k:getattr(ft,k) for k in ft.__dict__ 
                                    if not k.startswith('_')}
            })

    return template.render(**js)
