"""Decentralize Job scripts
"""
from __future__ import print_function
import argparse
import inspect
import sys
import json


def grep(txt, pattern):
    for line in txt.splitlines():
        if pattern in line:
            return line

def grepdoc(txt, item, prefix='* ', suffix=' :'):
    " return an item from documentation, for parameter help"
    pattern = prefix+item+suffix
    line = grep(txt, pattern)
    if not line:
        return None
    i = line.find(pattern)
    return line[i+len(pattern):].strip()


def add_optional_argument(parser, dest, default=None, aliases=None, required=False, help=None, **kwargs):
    """Add an optional argument to parser based on its dest name and default value

    (called by add_optional_argument)

    dest : same as `add_argument`
        The flag name `--ARG-NAME` or `--no-ARG-NAME` is built from `dest`.
    default : default value
        `type` and help is derived from default.
        If boolean, `store_true` or `store_false` will be determined, and 
        name updated.
    help : help text without default
    required : same as `add_argument`
    aliases : [str], additional aliases such as `--OTHER-NAME` or `-X`
    **kwargs keyword arguments are passed to `add_argument`
    """
    name = dest.replace('_','-')

    opt = {}

    if type(default) is bool:
        if default is False:
            opt["action"] = "store_true"
        else:
            opt["action"] = "store_false"
            name = 'no-'+name
    elif default is not None:
        opt["type"] = type(default)

    # determine help
    if default is not None and not required:
        if default is False:
            helpdef = '[default: False]'
        elif default is True:
            helpdef = '[default: True]'
        else:
            helpdef = '[default: %(default)s]'
        help = (help + " " + helpdef) if help else helpdef

    # update with user-specified
    opt.update(kwargs)

    return parser.add_argument('--'+name, *(aliases or ()), dest=dest, required=required, help=help, default=default, **opt)



def json_default(item):
    """`default` of json.dumps: returns serializable version of object
    """
    return json.loads(item.tojson())


class SubConfig(object):
    """Store global configuration parameters

    * flat : parameters must have distinct names
    * params must not start with _ , nor have some of the protected values: tojson, read, parser
    """

    _root = None

    # class internals
    # ===============
    def __iter__(self):
        #return iter(self.__dict__)
        for k in self.__dict__:
            if not k.startswith('_'):
                yield k

    def _group(self, cls):
        """instantiate another class
        """
        return cls(**{k:getattr(self, k) for k in self if k in cls()})


    # for subclassing (class composition approach)
    # --------------------------------------------
    @classmethod
    def _isunit(cls):
        "direct instance of SubConfig"
        return cls.__bases__ == (SubConfig,)

    @classmethod
    def _units_mro(cls, exclude_self=False):
        " return all parameter subgroups that compose the class "
        for supercls in inspect.getmro(cls):
            if not issubclass(supercls, SubConfig): continue
            if supercls._isunit() and (cls != supercls or not exclude_self):
                yield supercls

    @classmethod
    def _super_init(cls, self, **kwargs): 
        """dispatch arguments to the bases
        """
        # loop over super classes
        for supercls in cls._units_mro(exclude_self=True):
            opts = {k:kwargs.pop(k) for k in supercls() if k in kwargs}
            supercls.__init__(self, **opts)

        # nothing should be left over
        if kwargs:
            raise TypeError('invalid arguments: '+repr(kwargs))

    def _as_units(self, exclude_self=False):
        for supercls in self._units_mro(exclude_self):
            yield self._group(supercls)

    def _own(self):
        """dict of params that only belong to this class
        """
        parents = set()
        for grp in self._as_units(exclude_self=True):
            parents = parents.union(set(grp))
        return {k:getattr(self,k) for k in set(self).difference(parents)}


    # Parser tools
    # ============
    def _parser(self, add_help=False, **kwargs):
        " initiate parser"
        parents = [grp.parser for grp in self._as_units(exclude_self=True)]
        if not 'parents' in kwargs and parents:
            kwargs['parents'] = parents
        parser = argparse.ArgumentParser(add_help=add_help, **kwargs)
        return parser

    @classmethod
    def _doc(cls, name=None):
        " return doc about one param or the group"
        if not cls.__doc__:
            return ""
        if not name:  
            return cls.__doc__.splitlines()[0].capitalize()  # header
        return grepdoc(cls.__doc__, name)

    def _add_argument(self, parser, name, **kwargs):
        """add argument to a parser instance, based on diagnosed default
        """
        kwargs["default"] = kwargs.pop("default", getattr(self, name))
        kwargs["help"] = kwargs.pop("help", self._doc(name))
        return add_optional_argument(parser, name, **kwargs)

    def _add_argument_group(self, parser, nogroup=False):
        """feeling lucky: add parser group and all its arguments
        """
        own = self._own() # all params but no super classes !
        if not own:  # no own arguments
            return
            
        if nogroup:
            grp = parser
        else:
            grp = parser.add_argument_group(self._doc())

        for p in own: 
            self._add_argument(grp, p)
        return parser

    @classmethod
    def _defaults(cls):
        " dict of default parameters "
        return cls().__dict__

    def _parser_auto(self, nogroup=False, **kwargs):
        parser = self._parser(**kwargs)  # all parents
        self._add_argument_group(parser, nogroup)
        return parser

    @property
    def parser(self):
        return self._parser_auto()

    # I/O to json format
    # ==================
    def _partition(self):
        """partition of params as an iterator of tuples (root, **params)
        """
        # units
        for u in self._as_units():
            yield u._root, u.__dict__

        # anything that is not a direct subclass of SubConfig but does add own arguments
        own = self._own()
        if not self._isunit() and own:
            yield self._root, own
             

    def tojson(self, diff=False, flat=False, indent=2, sort_keys=True, **kwargs):
        """
        * diff : if True, only include differences to default
        * flat : if true, do not account for hierarchies (`_root` attribute)
        """
        _diff = lambda x, y: {k:x[k] for k in x if x[k] != y[k]}
        defaults = self._defaults()
        
        if flat:
            cfg = _diff(self.__dict__, defaults) if diff else self.__dict__
        else:
            cfg = {}
            for key, dict_ in self._partition():
                if diff:
                    dict_ = _diff(dict_, defaults)
                if key and dict_:
                    dict_ = {key : dict_}
                cfg.update(dict_)

        return json.dumps(cfg, indent=indent, sort_keys=sort_keys, default=json_default, **kwargs)

    @classmethod
    def read(cls, file, flat=False):
        """read subconfig from file
        """
        cfg = json.load(open(file))

        # saved without grouping?
        if flat:
            return cls(**{k:cfg[k] for k in cfg if k in cls()})

        kwargs = {}
        for key, dict_ in cls()._partition():
            if key:
                cfg_ = cfg.pop(key, {})
            else:
                cfg_ = cfg
            kwargs.update({k:cfg_[k] for k in cfg_ if k in dict_})

        return cls(**kwargs)

    def _update(self, cfg):
        """like update_known but raise an error if unknown args
        """
        unknown = self._update_known(cfg)
        if unknown:
            raise ValueError('{} :: unknown parameters: {}'.format( 
                self.__class__.__name__, 
                ", ".join(unknown.keys())))

    def _update_known(self, cfg):
        """pick whatever is known in cfg to update and return the rest
        """
        cfg = cfg.copy()
        for k in self:
            setattr(self, k, cfg.pop(k, getattr(self, k)))
        return cfg

    def _assert_internals(self):
        """check internal consistency

        * parser namespace match class namespace
        * instance and class methods are disjoint
        """
        namespace = self.parser.parse_args(['--executable', self.executable])
        not_in_self = set(namespace.__dict__).difference(set(self))
        assert not not_in_self, "unknown argument in parser: "+repr(not_in_self)
        not_in_parser = set(self).difference(set(namespace.__dict__))
        if not_in_parser:
            logging.warn(repr(type(cls))+":: argument not in parser: "+repr(not_in_parser))
        to_be_renamed = set(self).intersection(set(type(self).__dict__))
        assert not to_be_renamed, \
            "conflicting method and argument names:"+repr(to_be_renamed)




#def _add_bases_as_prop(cls):
#    """add the bases as properties, e.g. to do model.filetype
#    """
#    for supercls in cls.__bases__:
#        if supercls._root is None: 
#            continue
#        setattr(cls, supercls._root, property(lambda self: self._group(supercls)))





class Job(object):
    """Multiple main functions organized as sub-commands.
    """
    def __init__(self, dest="cmd", **kwargs):

        self.parser = argparse.ArgumentParser(**kwargs)
        self.subparsers = self.parser.add_subparsers(dest=dest)
        self.commands = {}
        self.dest = dest

    def add_command(self, name, command, **kwargs):
        """Register a command (`command`)
        """
        assert callable(command)
        subparser = self.subparsers.add_parser(name, **kwargs)
        self.commands[name] = command

    def main(self, argv=None):
        """Exectute program by calling the command
        """
        if argv is None:
            argv = sys.argv[1:]

        if '--debug' in argv:
            debug = True
            argv.remove('--debug') 
        else:
            debug = False

        # if subcommand, just call it:
        if len(argv) > 0:
            if argv[0] in self.commands:
                program = self.commands[argv[0]]
                sys.argv[0] = " ".join(sys.argv[:2]) # for help message
                try:
                    return program(argv[1:])
                except Exception as error:
                    if debug:
                        raise
                    print("ERROR:",error.message)
                    print("(use '--debug' for full traceback)")
                    sys.exit(1)


        # error message and help:
        self.parser.parse_args(argv)
