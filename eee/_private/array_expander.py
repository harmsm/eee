"""
Function for manipulating sets of arrays to keep the same length arrays. 
"""

import numpy as np

def array_expander(values):
    """
    Given a list or dict of input values (that could be a mixture of arrays and
    single values), return a list or dict of float arrays where each has the
    same length. Single values are expanded to be the length of the other
    arrays. If all entries are non-iterable, return the list or dict. 
    """

    # Is values a dictionary?
    if issubclass(type(values),dict):
        out = {}
        is_dict = True
    else:
        out = []
        is_dict = False

    # Does values contain an iterable?
    lengths_seen = []
    is_array = False
    for v in values:

        if is_dict:
            a = values[v]
        else:
            a = v

        if hasattr(a,"__iter__"):

            is_array = True
            lengths_seen.append(len(a))
    
    # If there was an iterable in values, expand all to be that length
    if is_array:

        # Multiple lengths seen -- throw an error
        if len(set(lengths_seen)) != 1:
            err = "All arrays must have the same length\n"
            raise ValueError(err)
        length = lengths_seen[0]

        # Go through each entry
        for v in values:

            if is_dict:
                a = values[v]
            else:
                a = v

            # If already an iterable, make sure its a float array; otherwise, 
            # create a new numpy array that is all the same value
            if hasattr(a,"__iter__"):
                new_value = np.array(a,dtype=float)
            else:
                new_value = a*np.ones(length,dtype=float)
            
            # Record results
            if is_dict:
                out[v] = new_value
            else:
                out.append(new_value)

    # If it was all floats, just return the float
    else:
        out = values
        length = 0

    return out, length
