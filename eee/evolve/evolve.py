
from eee.ensemble import Ensemble
from eee.evolve import wright_fisher
from eee.evolve.fitness import _fitness_function

import numpy as np
import pandas as pd

def _create_ddg_dict(ddg_df):
    """
    Convert a ddg_df dataframe in to a dictionary of the form:
    
    ddg_dict["site"]["mut"]["species"]
    """

    ddg_dict = {}
    for i in ddg_df.index:
        row = ddg_df.loc[i,:]
        
        site = row["site"]
        mut = row["mut"]
        if site not in ddg_dict:
            ddg_dict[site] = {}
            
        ddg_dict[site][mut] = {}
        for k in row.keys()[1:-2]:
            ddg_dict[site][mut][k] = row[k]

    return ddg_dict


class FitnessContainer:
    """
    Convenience class used in evolutionary simulations. Holds onto fixed 
    aspects of the fitness (ensemble, chemical potentials, fitness functions,
    and temperature), allowing us to calculate fitness given only the 
    mut_energy of a particular gentoype. 
    """
    def __init__(self,
                 ens,
                 mu_dict,
                 fitness_fcns,
                 select_on="fx_obs",
                 fitness_kwargs={},
                 T=298.15):
        """
        See eee.evolve.fitness_function docstring for information on arguments.
        """
        
        self._ens = ens
        self._mu_dict = mu_dict
        self._fitness_fcns = fitness_fcns
        self._select_on = select_on
        self._fitness_kwargs = fitness_kwargs
        self._T = T

    def fitness(self,mut_energy):
        """
        Calculate the fitness of a genotype with total mutational energies 
        given by mut_dict. Fitness is defined as the product of the fitness 
        in each of the conditions specified in mu_dict. 
        """
        
        F = _fitness_function(ens=self._ens,
                              mut_energy=mut_energy,
                              mu_dict=self._mu_dict,
                              fitness_fcns=self._fitness_fcns,
                              select_on=self._select_on,
                              fitness_kwargs=self._fitness_kwargs,
                              T=self._T)

        return np.prod(F)
    

def simulate_evolution(ens,
                       ddg_df,
                       mu_dict,
                       fitness_fcns,
                       select_on="fx_obs",
                       fitness_kwargs={},
                       T=298.15,
                       population_size=1000,
                       mutation_rate=0.01,
                       num_generations=100):
    """
    Simulate the evolution of a population where the fitness is determined by 
    the ensemble. 

    Parameters
    ----------
    ens : eee.Ensemble 
        initialized instance of an Ensemble class
    ddg_df : pandas.DataFrame
        pandas dataframe with columns holding 'mut', 'site' (i.e., the 21 in 
        A21P), 'is_wt' (whether or not the mutation is wildtype), and then 
        columns for the predicted ddG for each species. Generally should be
        created by util.load_ddg().
    mu_dict : dict, optional
        dictionary of chemical potentials. keys are the names of chemical
        potentials. Values are floats or arrays of floats. Any arrays 
        specified must have the same length. If a chemical potential is not
        specified in the dictionary, its value is set to 0. 
    fitness_fcns : list-like
        list of fitness functions to apply. There should be one fitness function
        for each condition specified in mu_dict. The first argument of each 
        function must be either fx_obs or dG_obs. Other keyword arguments can be
        specified in fitness_kwargs.
    select_on : str, default="fx_obs"
        observable to pass to fitness_fcns. Should be either fx_obs or dG_obs
    fitness_kwargs : dict, optional
        pass these keyword arguments to the fitness_fcn
    T : float, default=298.15
        temperature in Kelvin. This can be an array; if so, its length must
        match the length of the arrays specified in mu_dict. 
    population_size : int, default=1000
        population size for the simulation. Should be > 0.
    mutation_rate : float, default=0.01
        mutation rate for the simulation. Should be > 0.
    num_generations : int, default=100
        number of generations to run the simulation for

    Returns
    -------
    out : EvolutionResults
        The EvolutionResults class has four attributes. 

        + out.genotypes:  all genotypes seen (as Genotype instances).

        + out.trajectories: the trajectory taken by each genotype. Each entry
          is a list of steps taken to get to that genotype, where the steps are
          integer indexes pointing to out.genotype. A trajectory of [0,10,48]
          means the genotype started as out.genotypes[0], became
          out.genotypes[10], then finally out.genotypes[48].
          
        + out.fitnesses holds the absolute fitness of each genotype.

        + out.generations is a list of dictionaries, with one entry per step 
          of the simulation. For each generation, the dictionary keys are the
          genotypes (e.g., 0, 11, 48) and the values are the counts of those
          genotypes in that generation.
    """

    if not issubclass(type(ens),Ensemble):
        err = "ens should be an instance of the Ensemble class"
        raise ValueError(f"\n{err}\n\n")
    
    if not issubclass(type(ddg_df),pd.DataFrame):
        err = "ddg_df should be a pandas dataframe with mutational effects"
        raise ValueError(f"\n{err}\n\n")
    
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
    try:
        T = float(T)
        if T <= 0:
            raise ValueError
    except ValueError:
        err = "T (temperature in K) must be a float greater than zero."
        raise ValueError(f"\n{err}\n\n")
    
    # Population size
    try:
        population_size = int(population_size)
        if population_size < 1:
            raise ValueError
    except ValueError:
        err = "population_size must be an integer greater than or equal to 1"
        raise ValueError(f"\n{err}\n\n")

    # Mutation rate
    try:
        mutation_rate = float(mutation_rate)
        if mutation_rate <= 0:
            raise ValueError
    except ValueError:
        err = "mutation_rate must be a float greater than zero"
        raise ValueError(f"\n{err}\n\n")

    # Number of generations
    try:
        num_generations = int(num_generations)
        if num_generations < 1:
            raise ValueError
    except ValueError:
        err = "num_generations must be an integer greater than or equal to 1"
        raise ValueError(f"\n{err}\n\n")

    # Create a dictionary of mutational effects 
    ddg_dict = _create_ddg_dict(ddg_df)

    # Build a FitnessContainer object to calculate fitness values from the 
    # ensemble.
    fc = FitnessContainer(ens=ens,
                          mu_dict=mu_dict,
                          fitness_fcns=fitness_fcns,
                          select_on=select_on,
                          fitness_kwargs=fitness_kwargs,
                          T=T)
    
    # Run and return a Wright Fisher simulation.
    out =  wright_fisher(ens=ens,
                         ddg_dict=ddg_dict,
                         fc=fc,
                         population_size=population_size,
                         mutation_rate=mutation_rate,
                         num_generations=num_generations)
    
    return out


