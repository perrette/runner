"""2to3 compat 
"""
try:
    from builtins import str as basestring
except:
    basestring = basestring
