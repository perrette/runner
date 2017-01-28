import datetime
import json
from simtools import __version__

# job config I/O
# ==============
def _parser_defaults(parser):
    " parser default values "
    return {a.dest: a.default for a in parser._actions}


def _modified(kw, defaults):
    """return key-words that are different from default parser values
    """
    return {k:kw[k] for k in kw if k in defaults and kw[k] != defaults[k]}

def _filter(kw, after, diff=False, include_none=True):
    if diff:
        filtered = _modified(kw, after)
    else:
        filtered = {k:kw[k] for k in kw if k in after}
    if not include_none:
        filtered = {k:filtered[k] for k in filtered if filtered[k] is not None}
    return filtered


def json_config(cfg, parser=None, diff=False, name=None):
    if parser is None:
        defaults = cfg
    else:
        defaults = _filter(cfg, _parser_defaults(parser), diff, include_none=False)
    js = {
        'defaults': defaults,
        'version':__version__,
        'date':str(datetime.date.today()),
        'name':name,  # just as metadata
    }
    return json.dumps(js, indent=2, sort_keys=True, default=lambda x: str(x))

def write_config(cfg, file, parser=None, diff=False, name=None):
    string = json_config(cfg, parser, diff, name)
    with open(file, 'w') as f:
        f.write(string)

def load_config(file, parser):
    " the parser type must be used"
    js = json.load(open(file))["defaults"]
    for a in parser._actions:
        if a.dest in js and a.type is not None:
            if isinstance(js[a.dest], list):
                js[a.dest] = [a.type(e) for e in js[a.dest]]
            else:
                js[a.dest] = a.type(js[a.dest])
    return js
