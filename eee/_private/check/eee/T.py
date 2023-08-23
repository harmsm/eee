
from eee._private.check.standard import check_float

import numpy as np

def check_T(T,num_conditions):
    """
    Check the sanity of a temperature, returning a numpy array num_conditions 
    long holding T. 
    """

    if issubclass(type(T),type):
        err = f"\n T ({T}) cannot be a type\n\n"
        raise ValueError(err)

    if hasattr(T,"__iter__") and not issubclass(type(T),str):

        if issubclass(type(T),dict):
            err = "\nT cannot be a dictionary\n\n"
            raise ValueError(err)

        if len(T) != num_conditions:
            err = "\nT should be the same length as the number of conditions. "
            err += f"{len(T)} vs. {num_conditions})\n\n"
            raise ValueError(err)

        # Convert to floats
        T = list(T)
        for i in range(len(T)):
            T[i] = check_float(T[i],
                               variable_name=f"T[{i}]",
                               minimum_allowed=0,
                               minimum_inclusive=False)

        # Coerce to numpy array
        T = np.array(T,dtype=float)
    
    else:
        T = check_float(T,
                        variable_name="T",
                        minimum_allowed=0,
                        minimum_inclusive=False)
        T = T*np.ones(num_conditions,dtype=float)

    return T
