"""Tools
"""
def _create_dirtree(a,chunksize=2):
    """create a directory tree from a single, long name

    e.g. "12345" --> ["1", "23", "45"]
    """
    b = a[::-1]  # reverse
    i = 0
    l = []
    while i < len(b):
        l.append(b[i:i+chunksize])
        i += chunksize
    return [e[::-1] for e in l[::-1]]


def _short(name, value):
    '''Output short string representation of parameter and value.
       Used for automatic folder name generation.'''

    # Store the param value as a string
    # Remove the plus sign in front of exponent
    # Remove directory slashes, periods and trailing .nc from string values
    value = "%s" % (value)
    if "+" in value: value = value.replace('+','')

    if "/" in value: value = value.replace('/','')
    if ".." in value: value = value.replace('..','')
    if ".nc" in value: value = value.replace('.nc','')

    # Remove all vowels and underscores from parameter name
    name = name
    for letter in ['a','e','i','o','u','A','E','I','O','U','_']:
        name = name[0] + name[1:].replace(letter, '')

    return ".".join([name,value])


def autofolder(params):
    '''Given a list of (name, value) tuples,
       generate an appropriate folder name.
    '''
    parts = []

    for p in params:
        parts.append( _short(*p) )

    return '.'.join(parts)
