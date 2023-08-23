"""
Function for checking common eee variable sanity.
"""

from eee._private.check.standard import check_float
from eee._private.check.standard import check_int


def check_population_size(population_size):

    return check_int(value=population_size,
                     variable_name="population_size",
                     minimum_allowed=0,
                     minimum_inclusive=False)

def check_mutation_rate(mutation_rate):

    return check_float(value=mutation_rate,
                       variable_name="mutation_rate",
                       minimum_allowed=0,
                       minimum_inclusive=False)

def check_num_generations(num_generations):
    
    return check_int(value=num_generations,
                     variable_name="num_generations",
                     minimum_allowed=0,
                     minimum_inclusive=False)

def check_burn_in_generations(burn_in_generations):
    
    return check_int(value=burn_in_generations,
                     variable_name="burn_in_generations",
                     minimum_allowed=0,
                     minimum_inclusive=False)

def check_num_mutations(num_mutations):
    
    return check_int(value=num_mutations,
                     variable_name="num_mutations",
                     minimum_allowed=0,
                     minimum_inclusive=False)
