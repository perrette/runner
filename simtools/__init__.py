from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

# register addons
from .addons import register_filetype
from .addons.namelist import Namelist
register_filetype("namelist", Namelist)
