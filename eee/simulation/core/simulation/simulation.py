"""
Base class for running simulations on thermodynamic ensembles under 
different selective conditions. Must be sub-classed to be used.  
"""

from eee._private.check.ensemble import check_ensemble
from eee._private.check.standard import check_int

from eee.simulation.core.fitness import Fitness
from eee.simulation.core.genotype import Genotype

import numpy as np

import json
import os

class Simulation:
    """
    Base class for running simulations on thermodynamic ensembles under 
    different selective conditions. Must be sub-classed to be used.  
    """
    
    calc_type = None
    run = None

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

        # Build a Fitness object to calculate fitness values from the ensemble.
        self._fc = Fitness(ens=self._ens,
                           mu_dict=mu_dict,
                           fitness_fcns=fitness_fcns,
                           select_on=select_on,
                           select_on_folded=select_on_folded,
                           fitness_kwargs=fitness_kwargs,
                           T=T)
        
        # Build a Genotype object which manages the genotypes over the 
        # simulation
        self._gc = Genotype(ens=self._ens,
                            fitness_function=self._fc.fitness,
                            ddg_df=ddg_df,
                            choice_function=self._rng.choice)

        # Make sure the Simulation object is subclassed.
        if self.__class__ is Simulation:
            err = "\nOnly subclasses of Simulation should be used\n\n"
            raise NotImplementedError(err)

        # Make sure the subclass defines calc_type
        if self.calc_type is None:
            err = f"\nsubclasses of {super().__class__} must define the property\n"
            err += "`calc_type`. This should be a simple string. For example, \n"
            err += "`cool_sim` would allow eee to recognize 'cool_sim' as\n"
            err += "referring to this class in a simulation.json file.\n\n"
            raise NotImplementedError(err)

        # Make sure the subclass defines run
        if self.run is None:
            err = \
            """
            subclasses of SimulationContainer must define the :code:`run`"
            function. This should be a function that runs the actual
            simulation. The function should:

            1. Have its first argument be :code:`output_directory`. 

            2. Check the sanity of all input arguments using the 
               :code:`eee._private.check.standard` functions.

            3. Create a :code:`calc_params` dictionary that holds the names and
               values of all arguments passed to :code:`run`. The function 
               should pass this dictionary to :code:`self._prepare_calc` before
               doing its run.

            4. Write its outputs to files in its current working directory. 
               (These will automatically be stored in 'output_directory'.)

            5. Close out the calculation after running using 
               :code:`self._complete_calc.`

            6. Use the :code:`@run_cleanly` decorator to ensure sane behavior in 
               the event of a crash. 

            7. Have an informative docstring that uses the numpy docstring 
               format. 

            Example:

            .. code-block::python

                def run(self,output_directory,an_int_argument):
                    '''
                    Do something cool.
                    
                    Parameters
                    ----------
                    an_int_argument : int
                        a useful number
                    '''
                    
                    an_int_argument = check_int(an_int_argument,
                                                variable_name="an_int_argument",
                                                minimum_allowed=0)

                    calc_params = {"an_int_argument":an_int_argument}
                    
                    self._prepare_calc(output_directory=output_directory,
                                       calc_params=calc_params)
                    
                    # generates output file 'results.txt'
                    run_stuff_here() 

                    self._complete_calc()
                    
            """

            raise NotImplementedError(err)

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

        system_dict = self.system_params

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

    @property
    def system_params(self):
        """
        Dictionary of parameters describing system. This includes information
        about the ensemble, fitness calculation, and current genotypes. 
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

        return system_dict
    
    def get_calc_description(self,calc_kwargs=None):
        """
        Produce a pretty, human-readable string describing the calculation.

        Parameters
        ----------
        calc_kwargs : dict, optional
            dictionary of keyword arguments that will be passed to self.run

        Returns
        -------
        description : str
            description of the calculation
        """

        def _underline(some_string):
            length = len(some_string)
            return f"{some_string}\n{length*'-'}\n"
        
        out = []

        out.append(_underline(f"Running a '{self.calc_type}' calculation"))
        out.append(_underline("Ensemble properties:"))
        out.append(f"{self._ens.species_df}\n\n")

        out.append(_underline("Fitness calculation:"))
        out.append(f"Selecting on: {self._fc.select_on}")
        out.append(f"Selecting on folded: {self._fc.select_on_folded}")
        out.append(f"\nConditions and fitness functions:")
        out.append(f"{self._fc.condition_df}\n")

        if calc_kwargs is not None:
            out.append(_underline("Calculation parameters"))
            for k in calc_kwargs:
                out.append(f"{k}: {calc_kwargs[k]}")
            out.append(f"seed: {self._seed}")

        return "\n".join(out)

    def fitness_from_energy(self,genotype_energy):
        """
        Return fitness given the energy of a genotype. 

        Parameters
        ----------
        genotype_energy : numpy.ndarray or dict
            if numpy array, treat as the energetic effects on all species in the
            ensemble, ordered as in self.ens.species. If a dictionary, treat
            as a mut_dict where keys are species and values are energetic effects.

        Returns
        -------
        fitness : float
            fitness integrated across all conditions
        """

        if issubclass(type(genotype_energy),dict):
            genotype_energy = self.ens.mut_dict_to_array(genotype_energy)

        return np.prod(self._gc._fitness_function(genotype_energy))

    @property
    def ens(self):
        return self._ens

    @property
    def fc(self):
        return self._fc

    @property
    def gc(self):
        return self._gc