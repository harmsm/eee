"""
Functions for automatically building lists of and validating fitness functions
(e.g. ff_on = "on"), etc. 
"""
import inspect
from . import ff

def _get_ff_available():
    """
    Look in the ff.py module for functions with the name ff_XX and create 
    dictionaries that map between function identity and name.
    """

    available_ff = {}

    all_objs = inspect.getmembers(ff)

    for a in all_objs:
        if a[0].startswith("ff_") and callable(a[1]):
            fcn_str = "_".join(a[0].split("_")[1:])
            fcn = a[1]

            available_ff[fcn_str] = fcn

    return available_ff

def _map_fitness_fcn_to_string(value,return_as):
    """
    Take an input value and map back and forth between the name of that function
    and a pointer to the function.
    
    Parameters
    ----------
    value : function or str
        should be a fitness function or string name of a function
    return_as : str
        must be either "function" (return the function) or "string" (return 
        the string name of the function). 

    Returns
    -------
    result : function or str
        function or str from mapping, depending on return_as
    """
    
    str_to_fcn = FF_AVAILABLE
    fcn_to_str = dict([(str_to_fcn[s],s) for s in str_to_fcn])

    # If what came in is a function ...
    if callable(value):
        fcn = value
        if value in fcn_to_str:
            fcn_str = fcn_to_str[fcn]
        else:
            fcn_str = f"{fcn}"

    # If what came in is a string ...
    elif issubclass(type(value),str):
        fcn_str = value
        if value in str_to_fcn:
            fcn = str_to_fcn[fcn_str]
        else:
            fcn = None

    # If neither, die
    else:
        err = f"\nvalue '{value}' should be a fitness function or string\n\n"
        raise ValueError(err)

    # Check return_as argument
    if not issubclass(type(return_as),str) or issubclass(type(return_as),type):
        err = f"\nreturn_as ({return_as}) should be 'function' or 'string'\n\n"
        raise ValueError(err)

    # Return function if possible
    if return_as == "function":

        if fcn is None:
            err = f"fitness function string ('{fcn_str}') should be one of:\n"
            for k in str_to_fcn:
                err += f"    {k}\n"
            err += "\n"
            raise ValueError(err)
        
        return fcn
    
    # Return string. (If a function was passed in that is not in
    # FF_AVAILABLE, just return it's name. Allows a custom function
    # to be passed in). 
    elif return_as == "string":

        return fcn_str
    
    # Or die. 
    else:
        err = f"\nreturn_as ({return_as}) should be 'function' or 'string'\n\n"
        raise ValueError(err)


def check_fitness_fcns(fitness_fcns,
                       num_conditions,
                       return_as="function"):
    """
    Check the sanity of a fitness_functions, returning a list num_conditions 
    long holding fitness functions. return_as can be 'function' (meaning return
    the function itself) or 'string' (meaning return the string identifier of 
    the function). 
    """

    if issubclass(type(fitness_fcns),type):
        err = f"\nfitness_fcns '{fitness_fcns} should not be a type\n\n"
        raise ValueError(err)

    # If a single function, expand to a list of functions
    if not hasattr(fitness_fcns,"__iter__"):
        fitness_fcns = [fitness_fcns for _ in range(num_conditions)]

    for f in fitness_fcns:
        if issubclass(type(f),type):
            err = f"\nfitness_fcns entry '{f} should not be a type\n\n"
            raise ValueError(err)

    # Convert the fitness functions to callable functions if specified as 
    # strings (like "on", "off", and "neutral")
    parsed_fitness_fcns = []
    for f in fitness_fcns:
        new_f = _map_fitness_fcn_to_string(f,return_as="function")
        parsed_fitness_fcns.append(new_f)

    fitness_fcns = parsed_fitness_fcns[:]

    # Make sure all fitness_fcns can be called
    for f in fitness_fcns:
        if not callable(f):
            err = "All entries in fitness_fcns should be functions that take\n"
            err += "an ensemble observable as their first argument."
            raise ValueError(f"\n{err}\n\n")
    
    # Make sure fitness functions is the right length
    if num_conditions is not None:

        # ligand_length must match the length of fitnesss_fcns (one fitness per 
        # condition).
        if len(fitness_fcns) != num_conditions:
            err = "fitness should be the same length as the number of conditions\n"
            err += "in ligand_dict.\n"
            raise ValueError(err)

    
    if return_as == "function":
        out = fitness_fcns
    elif return_as == "string":

        out = []
        for f in fitness_fcns:
            out.append(_map_fitness_fcn_to_string(f,return_as="string"))
    else:
        err = f"\nreturn_as '{return_as}' not recognized. Should be 'function'\n"
        err += "or 'string'\n\n"
        raise ValueError(err)


    return out

FF_AVAILABLE = _get_ff_available()