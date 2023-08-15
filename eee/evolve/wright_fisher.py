"""
Run a Wright-Fisher simulation given an ensemble.
"""

from eee.evolve.genotype import GenotypeContainer
from eee.evolve.analysis import get_num_accumulated_mutations
from eee._private.interface import MockTqdm

from eee._private.check.eee_variables import check_num_generations
from eee._private.check.eee_variables import check_mutation_rate
from eee._private.check.eee_variables import check_population_size
from eee._private.check.eee_variables import check_num_mutations
from eee._private.check.standard import check_int

import numpy as np
from tqdm.auto import tqdm

import warnings
import pickle
import os
import glob

def _write_outputs(gc,
                   generations,
                   write_prefix,
                   write_counter,
                   num_write_digits,
                   final_dump=False):
    
    if write_prefix is not None:

        if final_dump:
            gen_to_write = [dict(zip(*g)) for g in generations[:]]
            generations = []
            keep_genotypes = None
        else:
            gen_to_write = [dict(zip(*g)) for g in generations[:-1]]
            generations = [generations[-1]]
            keep_genotypes = list(generations[-1][0])

        gen_fmt_string = "{:s}_generations_{:0" + f"{num_write_digits:d}" + "d}.pickle"
        gen_out_file = gen_fmt_string.format(write_prefix,write_counter)
        with open(gen_out_file,'wb') as f:
            pickle.dump(gen_to_write,f)
        
        # Write out the genotypes
        gc_filename = f"{write_prefix}_genotypes.csv"
        gc.dump_to_csv(filename=gc_filename,
                       keep_genotypes=keep_genotypes)


    return gc, generations

def wright_fisher(gc,
                  population,
                  mutation_rate,
                  num_generations,
                  num_mutations=None,
                  disable_status_bar=False,
                  write_prefix=None,
                  write_frequency=1000):
    """
    Run a Wright-Fisher simulation. This is a relatively low-level function. 
    Most users should probably call this via other simulation functions like
    simulate_evolution and simulate_tree. 
    
    Parameters
    ----------
    gc : GenotypeContainer
        container holding genotypes with either a wildtype sequence or a 
        sequences from a previous evolutionary simulation. This will bring in 
        everything we need to calculate fitness of each genotype. 
    population : dict, list-like, or int
        population for the simulation. Can be a dictionary of populations where
        keys are genotypes and values are populations. Can be an array where the
        length is the total population size and the values are the genotypes. 
        {5:2,4:1,0:2} is equivalent to [5,5,4,0,0]. If an integer, interpret as
        the population size. Create an initial population consisting of all 
        wildtype.
    mutation_rate : float
        Lambda for poisson distribution to select the number of genotypes to 
        mutate at each generation. Should be >= 0. 
    num_generations : int
        number of generations to run the simulation. Should be >= 1. If 
        num_mutations is specified, this is the *maximum* number of generations 
        allowed. 
    num_mutations : int, optional
        stop the simulation after the most frequent genotype has num_mutations 
        mutations. Should be >= 1. If specified, the simulation will run until
        either num_mutations is reached OR the simulation hits num_generations.
    disable_status_bar : bool, default=False
        turn off the tqdm status bar for the calculation 
    write_prefix : str, optional
        write output files during the run with this prefix. If not specified, 
        do not write files. If specified, gc and generations will be returned
        *empty* as their contents will have been written to lower memory 
        consumption.
    write_frequency : int, default=1000
        write the generations out every write_frequency generations. 
    
    Returns
    -------
    gc : GenotypeContainer
        updated GenotypeContainer holding all new genotypes that appeared over 
        the simulation
    generations : list
        list of dicts where each entry in the list is a generation. each dict
        holds the population at that generation with keys as genotypes and 
        values as populations. 
    """

    if not issubclass(type(gc),GenotypeContainer):
        err = "\ngc should be a GenotypeContainer instance.\n\n"
        raise ValueError(err)

    parse_err = "\npopulation should be a population dictionary, array of\n"
    parse_err += "genotype indexes, or a positive integer indicating the\n"
    parse_err += "population size.\n\n"

    if hasattr(population,"__iter__"):

        # Only instances allowed
        if issubclass(type(population),type):
            raise ValueError(parse_err)

        # If someone passes in something like {5:10,8:40,9:1}, where keys are 
        # genotype indexes and values are population size, expand to a list of 
        # genotypes
        if issubclass(type(population),dict):
            
            _population = []
            for p in population:
                _population.extend([p for _ in range(population[p])])
            population = _population
        
        if issubclass(type(population),str):
            population_size = check_int(population)
            population = np.zeros(population_size,dtype=int)

        # Make sure population is a numpy array, whether passed in by user as 
        # a list or from the list built above
        population = list(population)
        for i in range(len(population)):
            population[i] = check_int(value=population[i],
                                      variable_name="population[i]",
                                      minimum_allowed=0)

        population = np.array(population,dtype=int)
        population_size = len(population)

    else:

        # Get the population size
        try:
            population_size = int(population)
        except (ValueError,TypeError,OverflowError):
            raise ValueError(parse_err)
        
        # Build a population of all wildtype
        population = np.zeros(population_size,dtype=int)
    
    # Check variables
    population_size = check_population_size(population_size)
    mutation_rate = check_mutation_rate(mutation_rate)
    num_generations = check_num_generations(num_generations)
    if num_mutations is not None:
        num_mutations = check_num_mutations(num_mutations)
    if write_prefix is not None:
        write_prefix = f"{write_prefix}"
        write_frequency = check_int(value=write_frequency,
                                    variable_name=write_frequency,
                                    minimum_allowed=1)
    
    in_gc = set(list(gc.genotypes.keys()))
    in_pop = set(list(population))
    if not in_pop.issubset(in_gc):
        err = f"\npopulation has genotype(s) that are not in the gc object\n\n"
        raise ValueError(err)

    # Get the mutation rate
    expected_num_mutations = mutation_rate*population_size

    # Dictionary of genotype populations
    seen, counts = np.unique(population,return_counts=True)
    generations = [(seen,counts)]

    # Turn off status bar if requested
    if disable_status_bar:
        pbar = MockTqdm(total=num_generations-1)
    else:
        pbar = tqdm(total=num_generations-1)

    # Remove existing files
    if write_prefix is not None:
        to_remove = glob.glob(f"{write_prefix}*.pickle")
        to_remove.extend(glob.glob(f"{write_prefix}*.csv"))
        for f in to_remove:
            os.remove(f)
                         
    # Run the simulation
    with pbar:
    
        hit_target_num_mutations = False

        # Set up to write files
        write_counter = 0
        num_write_digits = int(f"{num_generations/write_frequency:e}".split("e")[1]) + 1
        if num_write_digits < 0:
            num_write_digits = 1

        # For all num_generations (first is starting population)
        for i in range(1,num_generations):

            # Get the probability of each genotype: its frequency times its 
            # relative fitness. Get the current genotypes and their counts from the
            # last generation recorded
            current_genotypes, counts = generations[-1]
            prob = np.array([gc.fitnesses[g] for g in current_genotypes])
            prob = prob*counts
            
            # If total prob is zero, give all equal weights. (edge case -- all 
            # genotypes equally terrible)
            if np.sum(prob) == 0:
                prob = np.ones(population_size)

            # Calculate relative probability
            prob = prob/np.sum(prob)

            # Select offspring, with replacement weighted by prob
            population = np.random.choice(current_genotypes,
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

                # Generate a new mutant with a new index, then store that new index
                # in the population. 
                new_index = gc.mutate(index=population[j])        
                population[j] = new_index

            # Record populations
            seen, counts = np.unique(population,return_counts=True)
            generations.append((seen,counts))

            # If we are checking for number of mutations, check to see what the 
            # number of mutations is in the most frequent genotype. If that has 
            # greater than or equal to num_mutations, break. 
            if num_mutations is not None:

                num_mutations_seen = get_num_accumulated_mutations(seen=seen,
                                                                   counts=counts,
                                                                   gc=gc)
                
                if num_mutations_seen >= num_mutations:
                    hit_target_num_mutations = True
                    break
            
            if i % write_frequency == 0:

                gc, generations = _write_outputs(gc=gc,
                                                 generations=generations,
                                                 write_prefix=write_prefix,
                                                 write_counter=write_counter,
                                                 num_write_digits=num_write_digits,
                                                 final_dump=False)

                write_counter += 1

            pbar.update(n=1)


    if write_prefix is not None:
        gc, generations = _write_outputs(gc=gc,
                                        generations=generations,
                                        write_prefix=write_prefix,
                                        write_counter=write_counter,
                                        num_write_digits=num_write_digits,
                                        final_dump=True)

    # Warn if we did not get all of the requested mutations
    if num_mutations is not None and not hit_target_num_mutations:
        
        seen = generations[-1][0]
        counts = generations[-1][1]
    
        num_accum = get_num_accumulated_mutations(seen=seen,
                                                  counts=counts,
                                                  gc=gc)
        
        w = f"\n\nDid not accumulate requested number of mutations after {num_generations}\n"
        w += f"generations. Accumulated {num_accum} of {num_mutations} requested.\n"
        w += "Try increasing num_generations and/or mutation_rate.\n\n"
        warnings.warn(w)

    generations = [dict(zip(*g)) for g in generations]

    return gc, generations
