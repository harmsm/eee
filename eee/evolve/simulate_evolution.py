"""
Simulate a simple evolutionary trajectory.
"""

from eee.evolve import wright_fisher
from eee.evolve.fitness import FitnessContainer
from eee.evolve.genotype import GenotypeContainer

from eee._private.check.ensemble import check_ensemble
from eee._private.check.eee_variables import check_ddg_df
from eee._private.check.eee_variables import check_mu_dict
from eee._private.check.eee_variables import check_fitness_fcns
from eee._private.check.eee_variables import check_calc_params

import numpy as np
import pandas as pd


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
    gc : GenotypeContainer
        This object holds information about the genotypes seen over the 
        simulation. Key attributes include out.genotypes (all genotypes seen
        as Genotype instances), out.trajectories (list of mutations that 
        occurred to get to the specific genotype), and out.fitnesses 
        (absolute fitness of each genotype). See the GenotypeContainer docstring
        for more details.
    generations : list
        list of dictionaries, with one entry per step of the simulation. For
        each generation, the dictionary keys are the genotypes (e.g., 0, 11, 48)
        and the values are the counts of those genotypes in that generation.
    """


    ens = check_ensemble(ens,check_obs=True)
    ddg_df = check_ddg_df(ddg_df)
    mu_dict = check_mu_dict(mu_dict)
    fitness_fcns = check_fitness_fcns(fitness_fcns,mu_dict=mu_dict)

    check_calc_params(T=T,
                      population_size=population_size,
                      mutation_rate=mutation_rate,
                      num_generations=num_generations)

    # Build a FitnessContainer object to calculate fitness values from the 
    # ensemble.
    fc = FitnessContainer(ens=ens,
                          mu_dict=mu_dict,
                          fitness_fcns=fitness_fcns,
                          select_on=select_on,
                          fitness_kwargs=fitness_kwargs,
                          T=T)
    
    # Build a GenotypeContainer object which manages the genotypes over the 
    # simulation
    gc = GenotypeContainer(fc=fc,
                           ddg_df=ddg_df)
    
    # Run and return a Wright Fisher simulation.
    gc, generations =  wright_fisher(gc=gc,
                                     population=population_size,
                                     mutation_rate=mutation_rate,
                                     num_generations=num_generations)
    
    return gc, generations






