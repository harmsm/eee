"""
Function for checking common eee variable sanity.
"""

from eee._private.check.standard import check_float
from eee._private.check.standard import check_int

import pandas as pd
import numpy as np

def check_mu_stoich(mu_stoich):
    """
    Check the sanity of mu_stoich.
    """

    if not issubclass(type(mu_stoich),dict):
        err = "mu_stoich should be a dictionary that keys chemical species to stoichiometry\n"
        raise ValueError(err)
    
    for mu in mu_stoich:
        mu_stoich[mu] = check_float(value=mu_stoich[mu],
                                    variable_name=f"mu_stoich['{mu}']",
                                    minimum_allowed=0,
                                    minimum_inclusive=True)
        
    return mu_stoich

def check_mu_dict(mu_dict):
    """
    Check the sanity of mu_dict. Returns a validated mu_dict and the length of
    the mu_dict conditions. 
    """

    # Should be a dictionary
    if not issubclass(type(mu_dict),dict) or issubclass(type(mu_dict),type):
        err = "mu_dict should be a dictionary that keys chemical species to chemical potential\n"
        raise ValueError(err)
    
    # Empty dict: allowed, just return
    if len(mu_dict) == 0:
        return mu_dict, 0

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

def check_mut_energy(mut_energy):

    # Make sure mut_energy is a dictionary of floats
    if not issubclass(type(mut_energy),dict) or issubclass(type(mut_energy),type):
        err = "mut_energy should be a dictionary that keys chemical species to effects of mutations\n"
        raise ValueError(err)
    
    for s in mut_energy:
        mut_energy[s] = check_float(value=mut_energy[s],
                                    variable_name=f"mut_energy['{s}']")
    
    return mut_energy

def check_ddg_df(ddg_df):

    if not issubclass(type(ddg_df),pd.DataFrame):
        err = "ddg_df should be a pandas dataframe with mutational effects"
        raise ValueError(f"\n{err}\n\n")
    
    return ddg_df

def check_fitness_fcns(fitness_fcns,num_conditions):
    """
    Check the sanity of a fitness_functions, returning a list num_conditions 
    long holding fitness functions. 
    """

    # If a single function, expand to a list of functions
    if callable(fitness_fcns) and not issubclass(type(fitness_fcns),type):
        fitness_fcns = [fitness_fcns for _ in range(num_conditions)]

    if not hasattr(fitness_fcns,"__iter__") or issubclass(type(fitness_fcns),type):
        err = "fitness_fcns must be a list of functions that take an ensemble\n"
        err += "observable as their first argument.\n"
        raise ValueError(err)

    # Make sure all fitness_fcns can be called
    for f in fitness_fcns:
        if not callable(f):
            err = "All entries in fitness_fcns should be functions that take\n"
            err += "an ensemble observable as their first argument."
            raise ValueError(f"\n{err}\n\n")
    
    # Make sure fitness functions is the right length
    if num_conditions is not None:

        # mu_length must match the length of fitnesss_fcns (one fitness per 
        # condition).
        if len(fitness_fcns) != num_conditions:
            err = "fitness should be the same length as the number of conditions\n"
            err += "in mu_dict.\n"
            raise ValueError(err)

    return fitness_fcns

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
