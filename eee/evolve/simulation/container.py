
from eee._private.check.eee_variables import check_num_generations
from eee._private.check.eee_variables import check_mutation_rate
from eee._private.check.eee_variables import check_population_size
from eee._private.check.ensemble import check_ensemble
from eee._private.check.standard import check_int
from eee._private.interface import run_cleanly

from eee.evolve import FitnessContainer
from eee.evolve import GenotypeContainer
from eee.evolve.wright_fisher import wright_fisher

import numpy as np

import json
import os

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
        seed : int, optional
            positive integer used to do reproducible simulations
        """
        
        # ddg_df, mu_dict, fitness_fcns, select_on, select_on_folded,
        # fitness_kwargs, and T all tested by FitnessContainer and 
        # GenotypeContainer -- just pass through. 

        # Check observable
        self._ens = check_ensemble(ens,check_obs=True)

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
            v = np.random.Generator(np.random.PCG64(self._seed)).random()
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

    @property
    def calc_type(self):
        return "wright_fisher"

    #     # Make sure the SimulationContainer container is subclassed -- on itself
    #     if self.__class__ is SimulationContainer:
    #         err = "\nOnly subclasses of SimulationContainer should be used\n\n"
    #         raise NotImplementedError(err)

    #     # Make sure the subclass implements calc_type and run
    #     self.calc_type
    #     self.run

    # @property
    # def calc_type(self):
    #     err = f"\nsubclasses of {super().__class__} must define the property\n"
    #     err += "`calc_type`. This should be a simple string. For example, \n"
    #     err += "`cool_sim` would allow eee to recognize 'cool_sim' as\n"
    #     err += "referring to this class in a simulation.json file.\n\n"
    #     raise NotImplementedError(err)

    # @run_cleanly
    # def run(self,*args,**kwargs):

    #     err = f"\nsubclasses of {super().__class__} must define the `run`\n"
    #     err += "function. This should be a function that runs the actual\n"
    #     err += "simulation. We recommend that the function:\n\n":
    #     err += "  1. Checks the sanity of all input arguments using the\n"
    #     err += "     eee._private.check.standard functions.\n"
    #     err += "  2. Takes an argument `output_directory` and then runs the\n"
    #     err += "     calculation in that directory.\n"
    #     err += "  3. Creates a `calc_params` dictionary that holds the names\n"
    #     err += "     and values of all input parameters. The function should\n"
    #     err += "     then pass this dictionary to self._prepare_calc.\n"
    #     err += "  4. Closes out the calculation after running using\n"
    #     err += "     self._complete_calc.\n"
    #     err += "  5. Uses the @run_cleanly decorator, which cleans up the run\n"
    #     err += "     in the case of a crash.\n\n"

    #     raise NotImplementedError(err)
    
    #     # parse arguments and put all of them into "calc_params" dictionary
    #     self._prepare_calc(output_directory,calc_params)

    #     # Actually do calculation here, writing out any important outputs. 

    #     self._compete_calc()


    def _prepare_calc(self,
                      output_directory,
                      calc_params):
        """
        Move into output_directory and write the simulation parameters into 
        json and csv files. 
        """

        # Check for existing directory and remove if requested
        if os.path.exists(output_directory):
            err = f"\noutput_directory ({output_directory}) already exists\n\n"
            raise FileExistsError(err)

        # Create output directory and change into it
        os.mkdir(output_directory)
        self._current_dir = os.getcwd()
        os.chdir(output_directory)

        # Write calc inputs (json and csv)
        self._write_calc_params(calc_params=calc_params)

    def _write_calc_params(self,calc_params={}):
        """
        Write a simulation.json and ddg.csv file describing the simulation
        parameters. 
        """

        system_dict = {}

        # get ensemble
        ens = self._ens.to_dict()
        for k in ens:
            system_dict[k] = ens[k]

        # Get FitnessContainer
        fc = self._fc.to_dict()
        for k in fc:
            system_dict[k] = fc[k]
        
        # Get GenotypeContainer
        gc = self._gc.to_dict()
        for k in gc:
            system_dict[k] = gc[k]

        system_dict["seed"] = self._seed

        # Write ddg file
        self._gc._ddg_df.to_csv("ddg.csv")
        system_dict["ddg_df"] = "ddg.csv"

        # Get self
        out = {"calc_type":self.calc_type,
               "system":system_dict,
               "calc_params":calc_params}
    
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
        with open("simulation.json",'w') as f:
            json.dump(out,f,indent=2)

    def _complete_calc(self):
        """
        Clean up after a calculation.
        """

        # Return to starting dir if recorded
        if hasattr(self,"_current_dir"):
            os.chdir(self._current_dir)

    @run_cleanly
    def run(self,
            output_directory="eee_sim",
            population_size=1000,
            mutation_rate=0.01,
            num_generations=100,
            write_prefix="eee_sim",
            write_frequency=1000):
        """
        Run a simulation and save files to an output directory.
        
        Parameters
        ----------
        output_directory : str, default="eee_sim"
            do simulation in this output directory
        population_size : int, default=1000
            population size for the simulation. Should be > 0.
        mutation_rate : float, default=0.01
            mutation rate for the simulation. Should be > 0.
        num_generations : int, default=100
            number of generations to run the simulation for
        write_prefix : str, default="eee_sim"
            write output files during the run with this prefix. 
        write_frequency : int, default=1000
            write the generations out every write_frequency generations. 
        """

        population_size = check_population_size(population_size)
        mutation_rate = check_mutation_rate(mutation_rate) 
        num_generations = check_num_generations(num_generations)
        write_prefix = f"{write_prefix}"

        write_frequency = check_int(value=write_frequency,
                                    variable_name="write_frequency",
                                    minimum_allowed=1)
    
        # Record the new keys
        calc_params = {}
        calc_params["population_size"] = population_size
        calc_params["mutation_rate"] = mutation_rate
        calc_params["num_generations"] = num_generations
        calc_params["write_prefix"] = write_prefix
        calc_params["write_frequency"] = write_frequency

        self._prepare_calc(output_directory=output_directory,
                           calc_params=calc_params)
        
        # Run and return a Wright Fisher simulation.
        self._gc, _ =  wright_fisher(gc=self._gc,
                                     population=population_size,
                                     mutation_rate=mutation_rate,
                                     num_generations=num_generations,
                                     write_prefix=write_prefix,
                                     write_frequency=write_frequency,
                                     rng=self._rng)
        
        self._complete_calc()


    