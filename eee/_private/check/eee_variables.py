"""
Function for checking common eee variable sanity.
"""

from eee._private.check.standard import check_float
from eee._private.check.standard import check_int

from eee._private.array_expander import array_expander

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
                                    minimum_inclusive=False)
        
    return mu_stoich

def check_mu_dict(mu_dict):
    """
    Check the sanity of mu_dict. 
    """

    # Should be a dictionary
    if not issubclass(type(mu_dict),dict) or issubclass(type(mu_dict),type):
        err = "mu_dict should be a dictionary that keys chemical species to chemical potential\n"
        raise ValueError(err)
    
    # Check each key...
    for mu in mu_dict:

        # Value should not be a type
        if issubclass(type(mu_dict[mu]),type):
            err = f"mu_dict['{mu}'] is a type not an instance"
            raise ValueError(err)

        # Value should not be a pandas data frame
        if issubclass(type(mu_dict[mu]),pd.DataFrame):
            err = f"mu_dict['{mu}'] cannot be a pandas DataFrame"
            raise ValueError(err)

        # If it is iterable, we have work to do to check
        if hasattr(mu_dict[mu],"__iter__"):

            # If a string, try to coerce into a float
            if issubclass(type(mu_dict[mu]),str):
                mu_dict[mu] = check_float(mu_dict[mu],
                                            variable_name=f"mu_dict['{mu}']")
                
            # If a dictionary, die
            elif issubclass(type(mu_dict[mu]),dict):
                err = f"mu_dict['{mu}'] must be a float or array"
                raise ValueError(err)
            
            # Probably okay -- make sure length is bigger than zero and then 
            # coerce to a numpy float array
            else:
                if len(mu_dict[mu]) == 0:
                    err = f"mu_dict['{mu}'] must have a length > 0"
                    raise ValueError(err)

                mu_dict[mu] = np.array(mu_dict[mu],dtype=float)
        else:

            # Single value float
            mu_dict[mu] = check_float(value=mu_dict[mu],
                                        variable_name=f"mu_dict['{mu}']")
    
    
    # Check for expandability
    mu_dict, _ = array_expander(mu_dict)

    return mu_dict

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

def check_fitness_fcns(fitness_fcns,mu_dict=None):

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
    if mu_dict is not None:
        if len(fitness_fcns) != len(mu_dict[list(mu_dict.keys())[0]]):
            err = "fitness should be the same length as the number of conditions\n"
            err += "in mu_dict.\n"
            raise ValueError(err)

    return fitness_fcns

def check_T(T):

    return check_float(value=T,
                       variable_name="T",
                       minimum_allowed=0,
                       minimum_inclusive=False)

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
