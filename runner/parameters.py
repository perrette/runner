""" Base class to describe a parameter
"""
from __future__ import print_function
import warnings
import json

class ParamNameError(ValueError):
    """
    """
    pass

class Param(object):
    """Parameter class. It is a light-weight, flexible class designed to store parameter information.

    The only required information is a parameter name.
    """

    def __init__(self, name, value=None, group=None, help=None, units=None, module=None):
        # TODO: remote `group` and `module` attributes, which should be dealt at a higher level...
        # This would simplify things, programatically (remove the need for a `key` property).
        self.name = name
        self.value = value
        self.group = group or None
        self.help = help.strip() if help else None # .e.g. "blabla ({units})"
        self.units = units or None
        self.module = module or None

    @property
    def key(self):
        """uniquely identify one parameter by its name, module, group
        """
        key = self.name
        if self.group:
            key = self.group+"."+key
        if self.module:
            key = self.module+":"+key
        return key

    def __eq__(self, other):
        """Equality between two parameters determines how `params.index(p)` or `p in params` work.
        It is based on parameter `key`, and possibly, module.
        """
        if not isinstance(other, Param): 
            raise TypeError("Expected Param, got: {}".format(type(other)))
        return (self.name == other.name) \
            and (self.module is None or other.module is None or self.module == other.module) \
            and (self.group is None or other.group is None or self.group == other.group)

    def __repr__(self):
        """Unambiguous representation of the class for programmers
        """
        fmt = []
        if self.module is not None: fmt.append("module={module!r:}")  # does not include since it is not part of the key
        if self.group is not None: fmt.append("group={group!r:}")
        if self.name is not None: fmt.append("name={name!r:}")
        if self.value is not None: fmt.append("value={value!r:}")
        fmt = "Param("+",".join(fmt)+")"
        return fmt.format(**self.__dict__)

    def __str__(self):
        """Readable representation of a parameter.
        """
        return "{} = {!r:}".format(self.key, self.value)


class Params(list):
    """ list of parameters
    """
    def __init__(self, *args):
        # make sure we have a list
        list.__init__(self, *args)
        for p in self:
            if not isinstance(p, Param):
                print(type(p),":",p)
                raise TypeError("Expected Param, got: {}".format(type(p)))

    def append(self, param):
        if not isinstance(param, Param):
            raise TypeError("Expected Param, got: {}".format(type(param)))
        list.append(self, param)

    def index(self, param):
        if not isinstance(param, Param):
            raise TypeError("Expected Param, got: {}".format(type(param)))
        try:
            return list.index(self, param)
        except ValueError:
            raise ParamNameError("Parameter {!r} {} is not in list".format(param.key, 
                             "(module: {!r})".format(param.module) if param.module is not None else "") )

    def extend(self, params):
        if not isinstance(params, Params):
            raise TypeError("Expected Params, got: {}".format(type(params)))
        list.extend(self, params)

    def to_json(self, indent=2, **kwargs):
        import json
        def par_dict(p):
            d = vars(p)
            return {k:d[k] for k in d if d[k] is not None}
        return json.dumps([par_dict(p) for p in self], indent=indent, **kwargs)

    @classmethod
    def from_json(cls, string):
        import json
        return cls([Param(**p) for p in json.loads(string)])

    # generic method to be overloaded, default to json
    parse = from_json
    format = to_json

    def write(self, filename, mode='w', **kwargs):
        with open(filename, mode) as f:
            f.write(self.format(**kwargs))

    @classmethod
    def read(cls, filename):
        with open(filename) as f:
            params = cls.parse(f.read())
        return params

    # Convenience mehotds
    # ===================
    # The methods below make it convenient to play with a list
    # of parameters, but this is nothing that cannot be achieved
    # in one or two lines using built-in python methods for list.
    # Generally they simply add more informative error messages
    # and perform additional checks that no duplicate elements 
    # are in the list, etc...

    def filter(self, **kwargs):
        # wrapper for built-in `filter` method
        def func(p):
            res = True
            for k in kwargs:
                res = res and getattr(p, k) == kwargs[k]
            return res
        return self.__class__(filter(func, self))

    def get(self, name, group=None):
        """return parameter by name, and possibly group
        will raise an error if not exactly one param is found
        >>> params.get('rho_i').value = 910
        >>> params.get('A', group="dynamics").help
        """
        i = self.index( Param(name, group=group) )
        return self[i]

    def set(self, name, value, group=None, **kwargs):
        """for compatibility with previous Alex version
        """
        p = self.get(name, group)
        p.value = value
        for k in kwargs:
            setattr(p, k, kwargs[k])

    def update(self, others, extend=False):
        """ update existing parameters (dict-like)
        """
        for p in others:
            if p not in self and extend:
                self.append(p)
            else:
                self[self.index(p)] = p  


