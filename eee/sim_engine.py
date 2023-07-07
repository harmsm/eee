"""
Functions for simulating evolution in the presence of ensemble epistasis. 
"""

from eee.ddg import create_ddg_dict

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import copy

def _fitness_function(ens,
                      mut_energy,
                      mu_dict,
                      fitness_fcns,
                      select_on,
                      fitness_kwargs,
                      T):
    """
    Private fitness function without error checking. Should be called via the
    public fitness_function for use in the API.
    """
    
    num_conditions = len(fitness_fcns)

    values = ens.get_obs(mut_energy=mut_energy,
                         mu_dict=mu_dict,
                         T=T)
    
    all_F = np.zeros(num_conditions)
    for i in range(num_conditions):
        all_F[i] = fitness_fcns[i](values[select_on].iloc[i],**fitness_kwargs)

    return all_F
       

def fitness_function(ens,
                     mut_energy,
                     mu_dict,
                     fitness_fcns,
                     select_on="fx_obs",
                     fitness_kwargs={},
                     T=298.15):
    """

    Parameters
    ----------
    ens : eee.Ensemble 
        initialized instance of an Ensemble class
    mut_energy : dict, optional
        dictionary holding effects of mutations on different species. Keys
        should be species names, values should be floats with mutational
        effects in energy units determined by the ensemble gas constant. 
        If a species is not in the dictionary, the mutational effect for 
        that species is set to zero. 
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

    Returns
    -------
    F : numpy.array
        float numpy array with one fitness per condition. 
    """

    num_conditions = len(mu_dict[list(mu_dict.keys())[0]])

    if len(fitness_fcns) != num_conditions:
        err = "fitness should be the same length as the number of conditions\n"
        err += "in mu_dict.\n"
        raise ValueError(err)

    for f in fitness_fcns:
        if not callable(f):
            err = "Elements of the fitness vector must all be functions that\n"
            err += "take the values specified in `select_on` as inputs and\n"
            err += "return the absolute fitness.\n"
            raise ValueError(err)
        
    if select_on not in ["fx_obs","dG_obs"]:
        err = "select_on should be either fx_obs or dG_obs\n"
        raise ValueError(err)ddg_dict

    return _fitness_function(ens=ens,
                             mut_energy=mut_energy,
                             mu_dict=mu_dict,
                             fitness_fcns=fitness_fcns,
                             select_on=select_on,
                             fitness_kwargs=fitness_kwargs,
                             T=T)

class FitnessContainer:

    def __init__(self,
                 ens,
                 fitness_fcns,
                 select_on="fx_obs",
                 fitness_kwargs={},
                 T=298.15):
        
        self._ens = ens
        self._mut_energy = fitness_fcns
        self._select_on = select_on
        self._fitness_kwargs = fitness_kwargs
        self._T = T

    def fitness(self,
                mut_energy,
                mu_dict):
        
        F = _fitness_function(ens=self._ens,
                              mut_energy=mut_energy,
                              mu_dict=mu_dict,
                              fitness_fcns=self._fitness_fcns,
                              select_on=self._select_on,
                              fitness_kwargs=self._fitness_kwargs,
                              T=self._T)

        return np.prod(F)


class Genotype:

    def __init__(self,
                 ens,
                 ddg_dict,
                 sites=None,
                 mutations=None,
                 mut_energy=None,
                 fitness=None):
        
        if sites is None:
            sites = []
        if mutations is None:
            mutations = []
        if fitness is None:
            fitness = 0.0

        self._sites = copy.deepcopy(sites)
        self._mutations = copy.deepcopy(mutations)
        self._fitness = fitness

        self._ens = ens
        self._ddg_dict = ddg_dict
        self._possible_sites = list(self._ddg_dict.keys())
        self._mutations_at_sites= [list(self._ddg_dict[s].keys())
                                   for s in self._possible_sites]

        # Update mut_energy
        if mut_energy is not None:
            self._mut_energy = copy.deepcopy(mut_energy)
        else:
            self._mut_energy = {}
            for s in self._ens.species:
                self._mut_energy[s] = 0
            

    def copy(self):
        """
        Return a copy of the Genotype instance.
        """

        # This will copy ens and ddg_dict as references, but make new instances
        # of the sites, mutations, mut_energy, and fitness attributes. 
        return Genotype(self._ens,
                        self._ddg_dict,
                        sites=self._sites,
                        mutations=self._mutations,
                        fitness=self._fitness)


    def mutate(self):

        # Mutation to revert before adding new one
        prev_mut = None

        # Randomly choose a site to mutate
        site = np.random.choice(self._possible_sites)

        # If the site was already mutated, we need to mutate site back to wt
        # before mutating to new genotype
        if site in self._sites:

            # Get site of mutation in genotype
            idx = self._sites.index(site)

            # Get mutation to revert before adding new one
            prev_mut = self._mutations[idx]

            # Subtract the energetic effect of the previous mutation
            for species in self._ddg_dict[site][prev_mut]:
                self._mut_energy[species] -= self._ddg_dict[site][prev_mut][species]

            # Remove the old mutation
            self._sites.remove(idx)
            self._mutations.remove(idx)

        # Choose what mutation to introduce. It must be different than the 
        # previous mutation at this site. 
        while True:
            mutation = np.random.choice(self._mutations_at_sites[site])
            if mutation != prev_mut:
                break

        # Update genotype with new site mutation, mutation made, 
        # trajectory step, overall energy, and fitness
        self._sites.append(site)
        self._mutations.append(mutation)

    


def engine(ens,
           ddg_df,
           mu_dict,
           fitness_fcns,
           select_on,
           fitness_kwargs,
           population_size,
           mutation_rate,
           num_generations,
           T=298.15):


    # Convert the ddg dataframe into a dictionary of the form: 
    # ddg_dict[site][mutation_at_site][conformation_in_ensemble]
    ddg_dict = create_ddg_dict(ddg_df)

    # Create wildtype genotype
    wt = Genotype(ens,ddg_dict)

    # This list holds all genotypes seen over the simulation. Each genotype 
    all_genotypes = [wt]

    # Get the mutation rate
    expected_num_mutations = mutation_rate*population_size

    # List of generations. Is list of dicts, where each list entry is a
    # generation. Each dict keys indexes in all_genotypes to the population of
    # that genotype at that generation. 
    generations = [{0:population_size}]

    # Current population as a vector with individual genotypes.
    pop_vector = np.zeros(population_size,dtype=int)
    for i in range(1,num_generations):

        # Get fitness values for all genotypes in the population
        prob = np.array([all_genotypes[g]["fitness"] for g in pop_vector])

        # If total prob is non-zero, weight populations. Otherwise, just take
        # populations
        if np.sum(prob) != 0:
            prob = pop_vector*prob
        else:
            prob = pop_vector

        # Calculate relative probability
        prob = prob/np.sum(prob)

        # Select offspring, with replacement weighted by prob
        pop_vector = np.random.choice(pop_vector,p=prob,replace=True)
        
        # Introduce mutations
        num_to_mutate = np.random.poisson(expected_num_mutations)

        # If we have a ridiculously high mutation rate, do not mutate each
        # genotype more than once.
        if num_to_mutate > population_size:
            num_to_mutate = population_size

        # Mutate first num_to_mutate population members
        for j in range(num_to_mutate):

            starting_genotype = pop_vector[j]
            new_genotype = copy.deepcopy(all_genotypes[starting_genotype])

            new_genotype["fitness"] = _fitness_function(ens=ens,
                                                        mut_energy=new_genotype["mut_energy"],
                                                        mu_dict=mu_dict,
                                                        fitness_fcns=fitness_fcns,
                                                        select_on=select_on,
                                                        fitness_kwargs=fitness_kwargs,
                                                        T=T)
            
            pop_vector[j] = len(all_genotypes)
            all_genotypes.append(new_genotype)
        
        # Record populations
        seen, counts = np.unique(pop_vector,return_counts=True)

        ## XX NOT RIGHT --> make trajectory
        compressed_pop.append(zip(seen,counts))
        



def wf_engine_python(pops,
                     mutation_rate,
                     fitness,
                     neighbor_slicer,
                     neighbors):
    """
    A python implementation of the Wright Fisher engine.

    This function should not be called directly. Instead, use wf_engine
    wrapper. Wrapper has argument docs and does argument sanity checking.
    """

    # If zero population, don't bother with simulation
    if np.sum(pops[0,:]) == 0:
        return pops

    # Get number of genoptypes, population size, and expected number of mutations
    # each generation
    num_genotypes = len(fitness)
    population_size = sum(pops[0,:])
    expected_num_mutations = mutation_rate*population_size
    num_generations = len(pops)

    indexes = np.arange(num_genotypes,dtype=int)
    for i in range(1,num_generations):

        # Look at non-zero genotypes
        mask = indexes[pops[i-1,:] != 0]
        local_fitness = fitness[mask]
        local_pop = pops[i-1,mask]

        # If all fitness are 0 for the populated genotypes, probability of
        # reproducing depends only on how often each genotype occurs.
        if np.sum(local_fitness) == 0:
            prob = local_pop

        # In most cases, reproduction probability is given by how many of each
        # genotype times its fitness
        else:
            prob = local_pop*local_fitness

        # Normalize prob
        prob = prob/np.sum(prob)

        # New population selected based on relative fitness
        new_pop = np.random.choice(mask,size=population_size,p=prob,replace=True)

        # Introduce mutations
        num_to_mutate = np.random.poisson(expected_num_mutations)

        # If we have a ridiculously high mutation rate, do not mutate each
        # genotype more than once.
        if num_to_mutate > population_size:
            num_to_mutate = population_size

        for j in range(num_to_mutate):
            k = new_pop[j]

            # If neighbor_slicer[k,0] == -1, this genotype *has* no neighbors.
            # Mutation should lead to self.
            if neighbor_slicer[k,0] != -1:
                a = neighbors[neighbor_slicer[k,0]:neighbor_slicer[k,1]]
                new_pop[j] = np.random.choice(a,size=1)[0]

        # Count how often each genotype occurs and store in pops array
        idx, counts = np.unique(new_pop,return_counts=True)
        pops[i,idx] = counts

    return pops