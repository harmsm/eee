"""
Simulate evolution of an ensemble along an evolutionary tree. 
"""
from eee.ensemble import Ensemble
from eee.evolve.fitness import FitnessContainer
from eee.evolve.genotype import GenotypeContainer
from eee.evolve import wright_fisher
from eee.evolve._helper import get_num_accumulated_mutations

import ete3

import numpy as np

def _simulate_branch(start_node,
                     end_node,
                     gc,
                     seq_length,
                     mutation_rate,
                     num_generations):
    
    starting_pop = start_node.generations[-1]

    # Get number of mutations to accumulate based on the branch length times 
    # the sequence length
    branch_length = start_node.get_distance(end_node)
    num_mutations = int(np.round(branch_length*seq_length,0))

    # Get the number of mutations at the starting node. The number to accumulate
    # over the branch is the start + the branch length. (This is the total number
    # of mutations that have accumulated, including reversions and multiple 
    # mutations at the same site). 
    num_mutations_start = get_num_accumulated_mutations(seen=list(starting_pop.keys()),
                                                        counts=list(starting_pop.values()),
                                                        gc=gc)
    
    num_mutations = num_mutations + num_mutations_start

    # Force at least one mutation to occur on the branch
    if num_mutations == 0:
        num_mutations = 1
    
    gc, generations = wright_fisher(gc,
                                    population=start_node.generations[-1],
                                    mutation_rate=mutation_rate,
                                    num_generations=num_generations,
                                    num_mutations=num_mutations)
    
    # record generations, discarding first one because that was the initial generation
    end_node.add_feature("generations",
                         generations[1:])


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
                  num_generations=100,
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


    sequence_length = len(gc.wt_sequence)
    
    root = tree.get_tree_root()
    root.add_feature("generations",generations[:])

    for n in tree.traverse(strategy="levelorder"):
        
        if not n.is_leaf():
            
            # Get descendants
            left, right = n.get_children()

            # Simulate evolution from n to left descendant. (implicitly updates
            # gc and left node)
            _simulate_branch(start_node=n,
                             end_node=left,
                             gc=gc,
                             seq_length=sequence_length,
                             mutation_rate=mutation_rate,
                             num_generations=num_generations)

            # Simulate evolution from n to right descendent. (implicitly updates
            # gc and right node)
            _simulate_branch(start_node=n,
                             end_node=right,
                             gc=gc,
                             seq_length=sequence_length,
                             mutation_rate=mutation_rate,
                             num_generations=num_generations)


    return gc, tree