"""
Simulate evolution of an ensemble along an evolutionary tree. 
"""
from eee.ensemble import Ensemble
from eee.evolve.fitness import FitnessContainer
from eee.evolve.genotype import GenotypeContainer
from eee.evolve import wright_fisher

import ete3

import numpy as np

def simulate_tree(ens,
                  ddg_df,
                  mu_dict,
                  fitness_fcns,
                  newick,
                  select_on="fx_obs",
                  fitness_kwargs={},
                  T=298.15,
                  population_size=1000,
                  mutation_rate=0.01,
                  burn_in_generations=10):

    tree = ete3.Tree(newick)

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
    

    # Burn in to generate initial population
    gc, generations =  wright_fisher(gc=gc,
                                     population=population_size,
                                     mutation_rate=mutation_rate,
                                     num_generations=burn_in_generations)

    
    root = tree.get_tree_root()
    root.add_feature("generations",generations[:])

    for n in tree.traverse(strategy="levelorder"):
        
        if not n.is_leaf():
            
            # Get descendants
            left, right = n.get_children()

            # Simulate evolution along the branch from n to left
            N = int(np.round(n.get_distance(left))) + 1
            gc, generations = wright_fisher(gc,
                                            population=n.generations[-1],
                                            mutation_rate=mutation_rate,
                                            num_generations=N)
            # record generations, discarding first one because that was the initial generation
            left.add_feature("generations",
                             generations[1:])

            # Simulate evolution along the branch from n to right
            N = int(np.round(n.get_distance(right))) + 1
            gc, generations = wright_fisher(gc,
                                            population=n.generations[-1],
                                            mutation_rate=mutation_rate,
                                            num_generations=N)
            # record generations, discarding first one because that was the initial generation
            right.add_feature("generations",
                              generations[1:])
            
    return gc, tree