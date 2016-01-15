"""Sampling parameters
"""

def combiner(a):
    """Combine lists in a systematic fashion

    Example
    -------
    >>> a = [[1,2],[3,4,5],[6]]
    [[1, 3, 6], 
     [2, 3, 6], 
     [1, 4, 6], 
     [2, 4, 6], 
     [1, 5, 6], 
     [2, 5, 6]]
    """
    r = [[]]
    for x in a:
        t = []
        for y in x:
            for i in r:
                t.append(i+[y])
        r = t

    return r


