import eee

from eee._private.check.ensemble import check_ensemble
from eee._private.check.eee_variables import check_num_generations
from eee._private.check.eee_variables import check_mutation_rate
from eee._private.check.eee_variables import check_population_size
from eee._private.check.standard import check_int
from eee._private.interface import run_cleanly

from eee.io import load_ddg

from eee.evolve import FitnessContainer
from eee.evolve import GenotypeContainer
from eee.evolve.wright_fisher import wright_fisher
from eee.evolve.fitness import get_fitness_function

import numpy as np

import json
import os
import shutil

def load_json(json_file,use_stored_seed=False):
    """
    Load a json file describing a simulation. This file must have the 
    following top-level keys:
    
        'ens': a dictionary of species describing the ensemble.
        'mu_dict': a dictionary indicating the chemical potentials
                    over which to do the simulation, 
        'fitness_fcns': the fitness functions to apply for each of the
                        conditions
        'ddg_df': spreadsheet file with the effects of mutations on each 
                    conformation in the ensemble. 
    
    Many other keys are permitted; see the documentation.

    Parameters
    ----------
    json_file : str
        json file to load
    use_stored_seed : bool, default=False
        The 'seed' key in the json file (if present) is ignored unless
        use_stored_seed is set to True. The only time to re-use the seed 
        is to restart a simulation or reproduce it exactly for testing 
        purposes. 
    """

    # Read json file
    with open(json_file) as f:
        run = json.load(f)
    
    # **********************************************************************
    # Goes into Ensemble
    # **********************************************************************

    # ---------------------------- ens -------------------------------------
    
    if "ens" not in run:
        err = "\njson must have 'ens' key in top level that defines the\n"
        err += "the thermodynamic ensemble.\n\n"
        raise ValueError(err)
    
    # ------------------------------ R -------------------------------------

    # Get gas constant
    if "R" in run:
        R = run["R"]
    else:
        # Get default from Ensemble class
        R = eee.Ensemble()._R

    # Create ensemble from entries.  
    ens = eee.Ensemble(R=R)
    for e in run["ens"]:
        ens.add_species(e,**run["ens"][e])
    ens = check_ensemble(ens,check_obs=True)

    # **********************************************************************
    # Goes into FitnessContainer
    # **********************************************************************

    # ---------------------------- mu_dict ---------------------------------
    
    if "mu_dict" not in run:
        err = "\njson must have 'mu_dict' key in top level that holds the\n"
        err += "chemical potential conditions over which to simulate\n"
        err += "evolution.\n\n"
        raise ValueError(err)
    
    mu_dict = run["mu_dict"]
    
    # ------------------------- fitness functions --------------------------
    
    if "fitness_fcns" not in run:
        err = "\njson must have 'fitness_fcns' key in top level that holds thel\n"
        err += "fitness functions to use for calculation.\n\n"
        raise ValueError(err)
    
    # Get fitness functions from the str on/of/neutral entries
    _fitness_fcns = run["fitness_fcns"]
    fitness_fcns = []
    for f in _fitness_fcns:
        fitness_fcns.append(get_fitness_function(f))
        
    # ------------------------- optional arguments -------------------------
        
    try:
        select_on = run["select_on"] 
    except KeyError:
        select_on = "fx_obs"

    try:
        select_on_folded = run["select_on_folded"]
    except KeyError:
        select_on_folded = True
    
    try:
        fitness_kwargs = run["fitness_kwargs"]
    except KeyError:
        fitness_kwargs = {}
    
    try:
        T = run["T"]
    except KeyError:
        T = 298.15
    
    # **********************************************************************
    # Goes into GenotypeContainer
    # **********************************************************************
    
    # ---------------------------- ddg_df ----------------------------------
    
    if "ddg_df" not in run:
        err = "\njson must have 'ddg_df' key in top level that points to a file\n"
        err += "containing the energetic effects of mutations on different species\n"
        err += "in the ensemble.\n\n"
        raise ValueError(err)
    
    ddg_df = run["ddg_df"]
    
    # Load ddg_df here so we don't have to keep track of the file when/if we
    # start a simulation in new directory
    ddg_df = load_ddg(ddg_df)
    
    # **********************************************************************
    # Goes into SimulationContainer
    # **********************************************************************

    try:
        population_size = run["population_size"]
    except KeyError:
        population_size = 1000
    
    try:
        mutation_rate = run["mutation_rate"]
    except KeyError:
        mutation_rate=0.01

    try:
        num_generations = run["num_generations"]
    except KeyError:
        num_generations = 100

    try:
        write_prefix = run["write_prefix"]
    except KeyError:
        write_prefix = "eee_sim"
    
    try:
        write_frequency = run["write_frequency"]
    except KeyError:
        write_frequency = 1000
    
    # ---------------------------- seed ----------------------------------

    seed = None
    if use_stored_seed and "seed" in run:
        seed = run["seed"]

    # **********************************************************************
    # Set up run with all of the relevant input values
    # **********************************************************************

    sc = SimulationContainer(ens=ens,
                             ddg_df=ddg_df,
                             mu_dict=mu_dict,
                             fitness_fcns=fitness_fcns,
                             select_on=select_on,
                             select_on_folded=select_on_folded,
                             fitness_kwargs=fitness_kwargs,
                             T=T,
                             population_size=population_size,
                             mutation_rate=mutation_rate,
                             num_generations=num_generations,
                             write_prefix=write_prefix,
                             write_frequency=write_frequency,
                             seed=seed)

    return sc


class SimulationContainer:
    """
    Class for running Wright-Fisher simulation on a thermodynamic ensemble. 
    """
    
    def __init__(self,
                 ens,
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
                 write_prefix="eee_sim",
                 write_frequency=1000,
                 seed=None):
        """
        Set up a simulation of the evolution of a population where the fitness
        is determined by the ensemble. 

        Parameters
        ----------
        ens : eee.Ensemble 
            initialized instance of an Ensemble class
        ddg_df : pandas.DataFrame
            pandas dataframe with columns holding 'mut', 'site' (i.e., the 21 in 
            A21P), and then columns for the predicted ddG for each species.
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
        write_prefix : str, default="eee_sim"
            write output files during the run with this prefix. If not specified, 
            do not write files. If specified, gc and generations will be returned
            *empty* as their contents will have been written to lower memory 
            consumption. 
        write_frequency : int, default=1000
            write the generations out every write_frequency generations. 
        seed : int, optional
            positive integer used to do reproducible simulations
        """
        
        # ddg_df, mu_dict, fitness_fcns, select_on, select_on_folded,
        # fitness_kwargs, and T all tested by FitnessContainer/GenotypeContainer.
        # Just pass through. 

        # Check and store other variables
        self._ens = check_ensemble(ens=ens,check_obs=True)
        self._population_size = check_population_size(population_size)
        self._mutation_rate = check_mutation_rate(mutation_rate) 
        self._num_generations = check_num_generations(num_generations)
        self._write_prefix = write_prefix

        # seed should be an integer > 0
        write_frequency = check_int(value=write_frequency,
                                    variable_name="write_frequency",
                                    minimum_allowed=1)
        self._write_frequency = write_frequency
        
        # seed should be an integer >= 0
        if seed is not None:
            seed = check_int(value=seed,
                             variable_name="seed",
                             minimum_allowed=0)
        self._seed = seed

        # If seed is None, generate a seed that we store, then use that stored
        # seed to generate a new random number generator. We should be able to
        # reproduce our results with this post hoc seed. 
        if self._seed is None:
            v = np.random.Generator(np.random.PCG64(seed)).random()
            self._seed = int(np.floor(v*1e12))

        # Create a random number generator object. 
        self._pcg64 = np.random.PCG64(self._seed)
        self._rng = np.random.Generator(self._pcg64)

        # Build a FitnessContainer object to calculate fitness values from the 
        # ensemble.
        self._fc = FitnessContainer(ens=self._ens,
                                    mu_dict=mu_dict,
                                    fitness_fcns=fitness_fcns,
                                    select_on=select_on,
                                    select_on_folded=select_on_folded,
                                    fitness_kwargs=fitness_kwargs,
                                    T=T)
        
        # Build a GenotypeContainer object which manages the genotypes over the 
        # simulation
        self._gc = GenotypeContainer(ens=self._ens,
                                     fitness_function=self._fc.fitness,
                                     ddg_df=ddg_df,
                                     choice_function=self._rng.choice)        

    @run_cleanly
    def run(self,output_directory,overwrite=False):
        """
        Run a simulation and save files to an output directory.
        
        Parameters
        ----------
        output_directory : str
            do simulation in this output directory
        overwrite : bool, default=False
            overwrite an existing output directory
        """

        # Check for existing directory and remove if requested
        if os.path.exists(output_directory):
            if not overwrite:
                err = f"\noutput_directory ({output_directory}) already exists\n\n"
                raise FileExistsError(err)
            else:
                shutil.rmdir(output_directory)

        # Create output directory and change into it
        os.mkdir(output_directory)
        current_dir = os.getcwd()
        os.chdir(output_directory)

        # Write csv file with mutation energies and json file with simulation 
        # parameters
        self._gc._ddg_df.to_csv("ddg.csv")
        self.write_json(json_file="simulation.json",
                        add_keys={"ddg_df":"ddg.csv"})
        
        # Run and return a Wright Fisher simulation.
        self._gc, _ =  wright_fisher(gc=self._gc,
                                     population=self._population_size,
                                     mutation_rate=self._mutation_rate,
                                     num_generations=self._num_generations,
                                     write_prefix=self._write_prefix,
                                     write_frequency=self._write_frequency,
                                     rng=self._rng)
        
        # Return to starting dir
        os.chdir(current_dir)

    
    def to_dict(self):
        """
        Return a json-able dictionary describing the calculation parameters.
        """

        out = {}
        to_write = ["population_size",
                    "mutation_rate",
                    "num_generations",
                    "write_prefix",
                    "write_frequency",
                    "seed"]
        for a in to_write:
            out[a] = self.__dict__[f"_{a}"]

        return out    


    def write_json(self,json_file,add_keys=None):
        """
        Write a json file describing the simulation parameters. 

        Parameters
        ----------
        json_file : str
            json file to write
        add_keys : dict, optional
            dictionary to append to the final json
        """

        # Get self
        out = self.to_dict()

        # get ensemble
        ens = self._ens.to_dict()
        for k in ens:
            out[k] = ens[k]

        # Get FitnessContainer
        fc = self._fc.to_dict()
        for k in fc:
            out[k] = fc[k]
        
        # Get GenotypeContainer
        gc = self._gc.to_dict()
        for k in gc:
            out[k] = gc[k]

        # Add extra keys passed in
        if add_keys is not None:
            for k in add_keys:
                out[k] = add_keys[k]
    
        def iteratively_remove_ndarray(d):
            """
            Convert np.ndarray to lists in the output dictionary so they are 
            json-able. 
            """

            for k, v in d.items():     

                v_type = type(v)   
                if issubclass(v_type, dict):
                    iteratively_remove_ndarray(v)
                else:            
                    if issubclass(v_type,np.ndarray):
                        d[k] = list(v)

        # Remove np.ndarray. 
        iteratively_remove_ndarray(out)

        # write json. 
        with open(json_file,'w') as f:
            json.dump(out,f,indent=2)
