

import pandas as pd
import numpy as np

def check_dataframe(value,variable_name=None):
    """
    Take a list of dictionaries and convert to a dataframe. The keys of the 
    dictionary are treated as columns; the values as """

    # Create helpful text for error message
    if variable_name is not None:
        err_base = f"{variable_name} {value}"
    else:
        err_base = f"{value}"
    
    # Check for types, which cause extreme wackiness (edge case)
    if issubclass(type(value),type):
        err = f"\n{err_base} cannot be a type\n\n"
        raise ValueError(err)
    
    # If already a dataframe, return a dataframe
    if issubclass(type(value),pd.DataFrame):
        return value

    # If a dictionary, try to convert this into a dataframe directly
    if issubclass(type(value),dict):
        return pd.DataFrame(value)

    # Must be an iterable at this point
    if not hasattr(value,"__iter__"):
        err = f"\n{err_base} must be a list of dicts or a dict\n"
        raise ValueError(err)
    
    # Build a column dictionary. 
    all_columns = {}
    for i, row in enumerate(value):
        if not issubclass(type(row),dict):
            err = f"\n{err_base} must be a list of dicts or a dict\n"
            raise ValueError(err)
        
        for key in row.keys():
            if key not in all_columns:
                all_columns[key] = [None for _ in range(len(value))]
            all_columns[key][i] = row[key]


    for c in all_columns:
        
        num_none = np.sum([v is None for v in all_columns[c]])
        if num_none == 0:
            continue
        elif num_none == len(value) - 1:
            v = [v for v in all_columns[c] if v is not None][0]
            all_columns[c] = [v for _ in range(len(value))]
        else:
            err = f"\n{err_base} has mis-matched numbers of values\n\n"
            raise ValueError(err)
        
    return pd.DataFrame(all_columns)
        
