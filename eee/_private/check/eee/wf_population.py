"""
Validate the a Wright-Fisher population argument.
"""

from eee._private.check.standard import check_int

import numpy as np

def check_wf_population(population):
    """
    Validate and expand a population input to a Wright Fisher simulation. Can be
    a dictionary of populations where keys are genotypes and values are
    populations. Can be an array where the length is the total population size
    and the values are the genotypes. {5:2,4:1,0:2} is equivalent to [5,5,4,0,0].
    If an integer, interpret as the population size. Create an initial
    population consisting of all wildtype (0).
    
    Parameters
    ----------
    population : dict or list-like or int
        population to validate
        
    Returns
    -------
    population : numpy.ndarray
        population array
    """

    parse_err = "\npopulation should be a population dictionary, array of\n"
    parse_err += "genotype indexes, or a positive integer indicating the\n"
    parse_err += "population size.\n\n"

    if hasattr(population,"__iter__"):

        # Only instances allowed
        if issubclass(type(population),type):
            raise ValueError(parse_err)

        # If someone passes in something like {5:10,8:40,9:1}, where keys are 
        # genotype indexes and values are population size, expand to a list of 
        # genotypes
        if issubclass(type(population),dict):
            
            _population = []
            for p in population:
                _population.extend([p for _ in range(population[p])])
            population = _population
        
        if issubclass(type(population),str):
            population_size = check_int(population)
            population = np.zeros(population_size,dtype=int)

        # Make sure population is a numpy array, whether passed in by user as 
        # a list or from the list built above
        population = list(population)
        for i in range(len(population)):
            population[i] = check_int(value=population[i],
                                      variable_name="population[i]",
                                      minimum_allowed=0)

        population = np.array(population,dtype=int)
        population_size = len(population)

    else:

        # Get the population size
        try:
            population_size = int(population)
        except (ValueError,TypeError,OverflowError):
            raise ValueError(parse_err)
        
        # Build a population of all wildtype
        population = np.zeros(population_size,dtype=int)

    if len(population) == 0:
        err = "\nPopulation size must be > 0\n\n"
        raise ValueError(err)

    return population