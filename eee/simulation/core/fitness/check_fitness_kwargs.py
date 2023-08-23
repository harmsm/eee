

def check_fitness_kwargs(fitness_kwargs,fitness_fcns=None):

    if fitness_kwargs is None:
        fitness_kwargs = {}

    # make sure it's a dictionary
    if not issubclass(type(fitness_kwargs),dict):
        err = "\nfitness_kwargs must be a dict that can be interpreted as\n"
        err += "keyword arguments by the fitness_functions.\n\n"
        raise ValueError(err)
    
    for k in fitness_kwargs:
        if not issubclass(type(k),str):
            err = f"\nThe fitness kwarg key '{k}' is not a string and cannot\n"
            err += "be interpreted as a function keyword argument.\n\n"
            raise ValueError(err)
    
    # Validate the fitness_kwargs against the fitness_fcns if requested. 
    if fitness_fcns is not None:
        for f in fitness_fcns:
            try:
                f(1,**fitness_kwargs)
            except TypeError:
                err = f"\nThe fitness function {f} cannot take the fitness_kwargs\n"
                err += f"given ({fitness_kwargs}).\n\n"
                raise ValueError(err)

    return fitness_kwargs