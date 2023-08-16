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
from eee._private.check.eee_variables import check_T
from eee._private.check.eee_variables import check_num_generations
from eee._private.check.eee_variables import check_mutation_rate
from eee._private.check.eee_variables import check_population_size
from eee._private.check.standard import check_int
from eee._private.check.standard import check_bool

def simulate_evolution(ens,
                       ddg_df,
                       mu_dict,
                       fitness_fcns,
                       select_on="fx_obs",
                       select_on_folded=True,
                       fitness_kwargs=None,
                       T=298.15,
                       population_size=1000,
                       mutation_rate=0.01,
                       num_generations=100,
                       write_prefix=None,
                       write_frequency=1000):
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
    select_on_folded : bool, default=True
        add selection for folded protein. 
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
    write_prefix : str, optional
        write output files during the run with this prefix. If not specified, 
        do not write files. If specified, gc and generations will be returned
        *empty* as their contents will have been written to lower memory 
        consumption. NOTE: if run from the command line, this will default to 
        eee_sim.
    write_frequency : int, default=1000
        write the generations out every write_frequency generations. 

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
    mu_dict, num_conditions = check_mu_dict(mu_dict)
    fitness_fcns = check_fitness_fcns(fitness_fcns,
                                      num_conditions=num_conditions)
    T = check_T(T,num_conditions=num_conditions)
    population_size = check_population_size(population_size)
    mutation_rate = check_mutation_rate(mutation_rate)
    num_generations = check_num_generations(num_generations)
    select_on_folded = check_bool(value=select_on_folded,
                                  variable_name="select_on_folded")

    # Check write_prefix and write_frequency
    if write_prefix is not None:
        write_prefix = f"{write_prefix}"
    write_frequency = check_int(value=write_frequency,
                                variable_name="write_frequency",
                                minimum_allowed=1)

    # Build a FitnessContainer object to calculate fitness values from the 
    # ensemble.
    fc = FitnessContainer(ens=ens,
                          mu_dict=mu_dict,
                          fitness_fcns=fitness_fcns,
                          select_on=select_on,
                          select_on_folded=select_on_folded,
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
                                     num_generations=num_generations,
                                     write_prefix=write_prefix,
                                     write_frequency=write_frequency)
    
    return gc, generations






