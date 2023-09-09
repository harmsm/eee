"""
Class for calculating fitness from an ensemble during an evolutionary simulation. 
"""

from eee._private.check.ensemble import check_ensemble
from eee._private.check.eee import check_T

from .check_fitness_fcns import check_fitness_fcns
from .check_fitness_kwargs import check_fitness_kwargs

from eee._private.check.eee import check_ligand_dict

from eee._private.check.standard import check_bool

import numpy as np
import pandas as pd

import copy

class Fitness:
    """
    Class used to calculate fitness of genotypes in evolutionary simulations.
    It holds fixed aspects of the fitness (ensemble, chemical potentials,
    fitness functions, and temperature), allowing us to quickly calculate
    fitness given only the energy of a particular gentoype. 

    Notes
    -----
    This class stores a private *copy* of ens. This is so the ensemble z-matrix 
    stays identical for all calculations, even if the user uses the initial 
    ensemble object for a calculation with a different ligand_dict after initialization
    of the FitnessContainer object. (That calculation triggers creation of new
    z-matrix, which would potentially change output observables). 
    """
    def __init__(self,
                 ens,
                 ligand_dict,
                 fitness_fcns,
                 select_on="fx_obs",
                 select_on_folded=True,
                 fitness_kwargs=None,
                 T=298.15):
        """
        Parameters
        ----------
        ens : eee.Ensemble 
            initialized instance of an Ensemble class
        ligand_dict : dict, optional
            dictionary of chemical potentials. keys are the names of chemical
            potentials. Values are floats or arrays of floats. Any arrays 
            specified must have the same length. If a chemical potential is not
            specified in the dictionary, its value is set to 0. 
        fitness_fcns : function or list
            fitness function(s) to apply. Should either be a single function or list 
            of functions. Functions should take value from "select_on" as their
            first argument and **fitness_kwargs as their remaining arguments. If a 
            list, the list must be the same length as the number of conditions in 
            ligand_dict. 
        select_on : str, default="fx_obs"
            observable to pass to fitness_fcns. Should be either fx_obs or dG_obs
        fitness_kwargs : dict, optional
            pass these keyword arguments to the fitness_fcn
        select_on_folded : bool, default=True
            In addition to selecting on select_on, multiply the fitness by the 
            fraction of the protein molecules that are folded. 
        T : float, default=298.15
            temperature in Kelvin. This can be an array; if so, its length must
            match the length of the arrays specified in ligand_dict. 
        """
        
        ens = check_ensemble(ens)
        ligand_dict, num_conditions = check_ligand_dict(ligand_dict)
        fitness_fcns = check_fitness_fcns(fitness_fcns,
                                          num_conditions=num_conditions,
                                          return_as="function")
        
        if not issubclass(type(select_on),str) or select_on not in ["fx_obs","dG_obs"]:
            err = "select_on should be either fx_obs or dG_obs\n"
            raise ValueError(err)

        fitness_kwargs = check_fitness_kwargs(fitness_kwargs,
                                              fitness_fcns=fitness_fcns)
        select_on_folded = check_bool(value=select_on_folded,
                                    variable_name="select_on_folded")

        T = check_T(T=T,num_conditions=num_conditions)

        self._ens = ens
        self._private_ens = copy.deepcopy(ens)
        self._ligand_dict = ligand_dict
        self._fitness_fcns = fitness_fcns
        self._select_on = select_on
        self._select_on_folded = select_on_folded
        self._fitness_kwargs = fitness_kwargs
        self._T = T

        self._fitness_fcns_strings = check_fitness_fcns(self._fitness_fcns,
                                                        num_conditions=num_conditions,
                                                        return_as="string")
                
        self._private_ens.read_ligand_dict(ligand_dict=self._ligand_dict)
        self._obs_function = self._private_ens.get_observable_function(self._select_on)
        self._num_conditions = len(self._fitness_fcns)
        self._F_array = np.zeros(self._num_conditions,dtype=float)

    def fitness(self,mut_energy_array):
        """
        Calculate the fitness of a genotype with total mutational energies 
        given by mut_energy_array. Fitness is defined as the product of the
        fitness  in each of the conditions specified in ligand_dict. 

        mut_energy_array : numpy.ndarray
            array holding the effects of mutations on energy. values should be
            in the order of ens.species 
        """
        
        values, fx_folded = self._obs_function(mut_energy_array=mut_energy_array,
                                               T=self._T)

        for i in range(self._num_conditions):
            self._F_array[i] = self._fitness_fcns[i](values[i],
                                                     **self._fitness_kwargs)
 
        if self._select_on_folded:
            self._F_array = self._F_array*fx_folded
 
        return self._F_array
    
    def to_dict(self):
        """
        Return a json-able dictionary describing the fitness parameters.
        """

        to_write = ["ligand_dict",
                    "select_on",
                    "select_on_folded",
                    "fitness_kwargs",
                    "T"]
        out = {}
        for a in to_write:
            out[a] = self.__dict__[f"_{a}"]

        # Fitness functions as strings
        out["fitness_fcns"] = self._fitness_fcns_strings

        return out

    @property
    def ens(self):
        return self._ens
        
    @property
    def ligand_dict(self):
        return self._ligand_dict
    
    @property
    def select_on(self):
        return self._select_on

    @property
    def select_on_folded(self):
        return self._select_on_folded

    @property
    def fitness_kwargs(self):
        return self._fitness_kwargs    
    
    @property
    def T(self):
        return self._T
    
    @property
    def condition_df(self):
        """
        Conditions as a pandas dataframe. 
        """

        to_df = {}
        for lig in self._ligand_dict:
            to_df[lig] = self._ligand_dict[lig]
        
        to_df["T"] = self._T
        to_df["ff"] = self._fitness_fcns_strings
    
        return pd.DataFrame(to_df)
    
        