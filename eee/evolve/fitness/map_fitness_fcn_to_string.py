"""
Functions for automatically creating maps between function names and functions
used for fitness calcs at specific conditions.
 """

def _construct_ff_dicts():
    """
    Look in the ff.py module for functions with the name ff_XX and create 
    dictionaries that map between function identity and name.
    """

    import inspect
    from . import ff

    all_objs = inspect.getmembers(ff)
    fitness_str_to_function = {}
    fitness_function_to_str = {}
    for a in all_objs:
        if a[0].startswith("ff_") and callable(a[1]):
            fcn_str = "_".join(a[0].split("_")[1:])
            fcn = a[1]

            fitness_str_to_function[fcn_str] = fcn
            fitness_function_to_str[fcn] = fcn_str

    return fitness_str_to_function, fitness_function_to_str


_FITNESS_STR_TO_FUNCTION, _FITNESS_FUNCTION_TO_STR = _construct_ff_dicts()
ff_list = list(_FITNESS_STR_TO_FUNCTION.keys())
ff_list.sort()


def map_fitness_fcn_to_string(value,return_as):
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
    
    # If what came in is a function ...
    if callable(value):
        fcn = value
        if value in _FITNESS_FUNCTION_TO_STR:
            fcn_str = _FITNESS_FUNCTION_TO_STR[fcn]
        else:
            fcn_str = f"{fcn}"

    # If what came in is a string ...
    elif issubclass(type(value),str):
        fcn_str = value
        if value in _FITNESS_STR_TO_FUNCTION:
            fcn = _FITNESS_STR_TO_FUNCTION[fcn_str]
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
            for k in _FITNESS_STR_TO_FUNCTION:
                err += f"    {k}\n"
            err += "\n"
            raise ValueError(err)
        
        return fcn
    
    # Return string. (If a function was passed in that is not in
    # _FITNESS_FUNCTION_TO_STR, just return it's name. Allows a custom function
    # to be passed in). 
    elif return_as == "string":

        return fcn_str
    
    # Or die. 
    else:
        err = f"\nreturn_as ({return_as}) should be 'function' or 'string'\n\n"
        raise ValueError(err)
