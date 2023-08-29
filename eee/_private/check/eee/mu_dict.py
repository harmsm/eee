"""
Check the sanity of mu_dict. Returns a validated mu_dict and the length of
the mu_dict conditions. 
"""

from eee._private.check.standard import check_float

import numpy as np
import pandas as pd

def check_mu_dict(mu_dict):
    """
    Check the sanity of mu_dict, making sure all values are arrays with the 
    same length.
    
    Parameters
    ----------
    mu_dict : dict
        dictionary holding chemical potentials keyed to species name

    Returns
    -------
    mu_dict : dict
        validated mu_dict, where all values are arrays of the same length
    length : int
        the length of the mu_dict conditions
    """

    # Should be a dictionary
    if not issubclass(type(mu_dict),dict) or issubclass(type(mu_dict),type):
        err = "mu_dict should be a dictionary that keys chemical species to chemical potential\n"
        raise ValueError(err)
    
    # Empty dict: allowed, just return
    if len(mu_dict) == 0:
        return mu_dict, 1

    # Check each value...
    mu_lengths = []
    for mu in mu_dict:

        # Make sure not disallowed value class that has __iter__
        value_type = type(mu_dict[mu])
        for bad in [type,pd.DataFrame,dict]:
            if issubclass(value_type,bad):
                err = f"\nmu_dict['{mu}'] cannot be type {bad}\n\n"
                raise ValueError(err)
        
        # If it is iterable, we have work to do to check types
        if hasattr(mu_dict[mu],"__iter__"):

            # Make sure there is actually something in the iterable
            if len(mu_dict[mu]) == 0:
                err = f"\nmu_dict['{mu}'] must have a length > 0\n\n"
                raise ValueError(err)

            # If a string, try to coerce into a float.
            if issubclass(value_type,str):
                v = check_float(mu_dict[mu],variable_name=f"mu_dict['{mu}']")
                mu_dict[mu] = np.ones(1,dtype=float)*v
            
            # If we get here, coerce into a numpy array.
            else:
                mu_dict[mu] = np.array(mu_dict[mu],dtype=float)
                if len(mu_dict[mu]) != 1:
                    mu_lengths.append(len(mu_dict[mu]))

        else:

            # Single value float
            v = check_float(value=mu_dict[mu],
                            variable_name=f"mu_dict['{mu}']") 
            mu_dict[mu] = np.ones(1,dtype=float)*v

    # All lengths are 1: return
    if len(mu_lengths) == 0:
        return mu_dict, 1

    # Unique non-one mu_lengths
    mu_lengths = list(set(mu_lengths))
    if len(mu_lengths) > 1:
        err = "\nall values in mu_dict must have the same length\n\n"
        raise ValueError(err)
    
    # Take any values with length one and make them the same length as the 
    # longer value. 
    final_mu_length = mu_lengths[0]
    for mu in mu_dict:
        if len(mu_dict[mu]) == 1:
            mu_dict[mu] = np.ones(final_mu_length,dtype=float)*mu_dict[mu][0]

    return mu_dict, final_mu_length