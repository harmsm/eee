"""
Functions to run a Wright-Fisher simulation given an ensemble.
"""

from eee.evolve import Genotype

import numpy as np
from tqdm.auto import tqdm

class EvolutionResults:
    """
    Container class with attributes holding results of the simulation. 
    """

    def __init__(self,
                 genotypes,
                 trajectories,
                 fitnesses,
                 generations):
        
        self.genotypes = genotypes
        self.trajectories = trajectories
        self.fitnesses = fitnesses
        self.generations = generations


def wright_fisher(ens,
                  ddg_dict,
                  fc,
                  population_size,
                  mutation_rate,
                  num_generations):
    """
    
    """

    # genotypes: list of genotypes seen over simulation. entries are 
    #                instances of Genotype class
    # trajectories : list of list. each entry is a genotype (matches 
    #                genotypes) showing the evolutionary history of that 
    #                genotype
    # fitnesses : list of float. fitness for each genotype (matches 
    #             genotypes)
    # generations: list of dicts, where each list entry is a generation. Each
    #              dict keys indexes in genotypes to the population of
    #              that genotype at that generation. 


    # Convert the ddg dataframe into a dictionary of the form: 
    # ddg_dict[site][mutation_at_site][conformation_in_ensemble]

    # Create wildtype genotype
    wt = Genotype(ens,ddg_dict)

    # Get the mutation rate
    expected_num_mutations = mutation_rate*population_size

    # Lists of genotypes trajectories, and fitnesses. These all have the same
    # index scheme
    genotypes = [wt]
    trajectories = [[0]]
    fitnesses = [fc.fitness(wt.mut_energy)]

    # Dictionary of genotype populations
    generations = [{0:population_size}]

    # Current population as a vector with individual genotypes.
    current_pop = np.zeros(population_size,dtype=int)
    for i in tqdm(range(1,num_generations)):

        # Get fitness values for all genotypes in the population
        prob = np.array([fitnesses[g] for g in current_pop])
        
        # If total prob is zero, give all equal weights. (edge case -- all 
        # genotypes equally terrible)
        if np.sum(prob) == 0:
            prob = np.ones(population_size)

        # Calculate relative probability
        prob = prob/np.sum(prob)

        # Select offspring, with replacement weighted by prob
        current_pop = np.random.choice(current_pop,
                                       size=population_size,
                                       p=prob,
                                       replace=True)
        
        # Introduce mutations
        num_to_mutate = np.random.poisson(expected_num_mutations)

        # If we have a ridiculously high mutation rate, do not mutate each
        # genotype more than once.
        if num_to_mutate > population_size:
            num_to_mutate = population_size

        # Mutate first num_to_mutate population members
        for j in range(num_to_mutate):

            # Genotype to mutate
            old_genotype = genotypes[current_pop[j]]

            # Create a new genotype and mutate
            new_genotype = old_genotype.copy()
            new_genotype.mutate()
            
            # Get index for new genotype
            new_index = len(genotypes)

            # Replace genotype "j" with the index of the new genotype in the 
            # population            
            current_pop[j] = new_index

            # Record the new genotype
            genotypes.append(new_genotype)

            # Record the trajectory of the current genotype
            new_trajectory = trajectories[j][:]
            new_trajectory.append(new_index)
            trajectories.append(new_trajectory)
            
            fitnesses.append(fc.fitness(new_genotype.mut_energy))

        # Record populations
        seen, counts = np.unique(current_pop,return_counts=True)
        generations.append(dict(zip(seen,counts)))

    return EvolutionResults(genotypes=genotypes,
                            trajectories=trajectories,
                            fitnesses=fitnesses,
                            generations=generations)
   
        

