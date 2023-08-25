"""
Function for checking common eee variable sanity.
"""

from eee._private.check.standard import check_float
from eee._private.check.standard import check_int


def check_population_size(population_size):
    """
    Validate population size (positive integer).
    
    Parameters
    ----------
    population_size : int
        population size for a simulation
    
    Returns
    -------
    population_size : int
        validated population size
    """

    return check_int(value=population_size,
                     variable_name="population_size",
                     minimum_allowed=0,
                     minimum_inclusive=False)

def check_mutation_rate(mutation_rate):
    """
    Validate mutation rate (positive float).
    
    Parameters
    ----------
    mutation_rate : float
        mutation rate for a simulation
    
    Returns
    -------
    mutation_rate : float
        validated mutation rate
    """

    return check_float(value=mutation_rate,
                       variable_name="mutation_rate",
                       minimum_allowed=0,
                       minimum_inclusive=False)

def check_num_generations(num_generations):
    """
    Validate number of generations (positive integer).
    
    Parameters
    ----------
    num_generations : int
        number of generations for a simulation
    
    Returns
    -------
    num_generations : int
        validated number of generations
    """
    
    return check_int(value=num_generations,
                     variable_name="num_generations",
                     minimum_allowed=0,
                     minimum_inclusive=True)

def check_burn_in_generations(burn_in_generations):
    """
    Validate number of burn in generations (positive integer or zero).
    
    Parameters
    ----------
    burn_in_generations : int
        number of burn in generations for a simulation
    
    Returns
    -------
    burn_in_generations : int
        validated number of burn in generations
    """
    
    return check_int(value=burn_in_generations,
                     variable_name="burn_in_generations",
                     minimum_allowed=0,
                     minimum_inclusive=True)

def check_num_mutations(num_mutations):
    """
    Validate target number of mutations (positive integer).
    
    Parameters
    ----------
    num_mutations : int
        number of target number of mutations for a simulation
    
    Returns
    -------
    num_mutations : int
        validated target number of mutations
    """
    
    return check_int(value=num_mutations,
                     variable_name="num_mutations",
                     minimum_allowed=0,
                     minimum_inclusive=False)
