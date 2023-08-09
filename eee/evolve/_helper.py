from eee.ensemble import Ensemble

import pandas as pd
import numpy as np

def get_num_accumulated_mutations(gc,
                                  seen,
                                  counts):
    """
    Get the total number of mutations that have accumulated (including multiple
    mutations at the same site) for the most frequent genotype in the population.
    """

    to_sort = list(zip(counts,seen))
    to_sort.sort()
    genotype = to_sort[-1][1]
    num_mutations = len(gc.genotypes[genotype].mutations_accumulated)

    return num_mutations

class MockTqdm():
    """
    Fake tqdm progress bar so we don't have to show a status bar if we don't
    want to. Can be substituted wherever we would use tqdm (i.e.
    tqdm(range(10)) --> MockTqdm(range(10)).
    """

    def __init__(self,*args,**kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self, type, value, traceback):
        pass

    def update(self,*args,**kwargs):
        pass

def check_arg_sanity(ens=None,
                     ddg_df=None,
                     mu_dict=None,
                     fitness_fcns=None,
                     T=None,
                     population_size=None,
                     mutation_rate=None,
                     num_generations=None,
                     burn_in_generations=None):
 

    if ens is not None:
        if not issubclass(type(ens),Ensemble):
            err = "ens should be an instance of the Ensemble class"
            raise ValueError(f"\n{err}\n\n")
    
    if ddg_df is not None:
        if not issubclass(type(ddg_df),pd.DataFrame):
            err = "ddg_df should be a pandas dataframe with mutational effects"
            raise ValueError(f"\n{err}\n\n")
    
    if mu_dict is not None:
        if not issubclass(type(mu_dict),dict):
            err = "mu_dict should be a dictionary of chemical potentials"
            raise ValueError(f"\n{err}\n\n")

        # Check lengths of mu_dict
        mu_lengths = []
        for m in mu_dict:
            mu_lengths.append(len(mu_dict[m]))
        mu_lengths = np.unique(mu_lengths)
        if len(mu_lengths) > 1:
            err = "All entries in mu_dict must have the same length."
            raise ValueError(f"\n{err}\n\n")
    
    if fitness_fcns is not None:

        # Check length of fitness_fcns
        if len(fitness_fcns) != mu_lengths[0]:
            err = "There must be an entry in fitness_fcns for each condition in\n "
            err += "mu_dict."
            raise ValueError(f"\n{err}\n\n")
    
        # Make sure all fitness_fcns can be called
        for f in fitness_fcns:
            if not callable(f):
                err = "All entries in fitness_fcns should be functions that take\n"
                err += "an ensemble observable as their first argument."
                raise ValueError(f"\n{err}\n\n")
    
    # Temperature
    if T is not None:
        
        try:
            T = float(T)
            if T <= 0:
                raise ValueError
        except ValueError:
            err = "T (temperature in K) must be a float greater than zero."
            raise ValueError(f"\n{err}\n\n")
        
    # Population size
    if population_size is not None:

        try:
            population_size = int(population_size)
            if population_size < 1:
                raise ValueError
        except ValueError:
            err = "population_size must be an integer greater than or equal to 1"
            raise ValueError(f"\n{err}\n\n")

    # Mutation rate
    if mutation_rate is not None:

        try:
            mutation_rate = float(mutation_rate)
            if mutation_rate <= 0:
                raise ValueError
        except ValueError:
            err = "mutation_rate must be a float greater than zero"
            raise ValueError(f"\n{err}\n\n")

    # Number of generations
    if num_generations is not None:

        try:
            num_generations = int(num_generations)
            if num_generations < 1:
                raise ValueError
        except ValueError:
            err = "num_generations must be an integer greater than or equal to 1"
            raise ValueError(f"\n{err}\n\n")

    # Number of generations
    if burn_in_generations is not None:

        try:
            burn_in_generations = int(burn_in_generations)
            if burn_in_generations < 1:
                raise ValueError
        except ValueError:
            err = "burn_in_generations must be an integer greater than or equal to 1"
            raise ValueError(f"\n{err}\n\n")
