"""
Functions and classes for calculating fitness from an ensemble during an
evolutionary simulation. 
"""

from eee._private.check.eee_variables import check_T
from eee._private.check.ensemble import check_ensemble
from eee._private.check.eee_variables import check_fitness_fcns
from eee._private.check.eee_variables import check_fitness_kwargs
from eee._private.check.eee_variables import check_mu_dict
from eee._private.check.eee_variables import check_mut_energy
from eee._private.check.standard import check_bool

import numpy as np

import copy

def ff_on(value):
    """
    Microscopic fitness function for fx_obs. Fitness is linearly proportional
    to value. Useful for simulating selection to keep observable 'on'.

    string_name: on
    """
    return value

def ff_off(value):
    """
    Microscopic fitness function for fx_obs. Fitness is linearly proportional
    to 1 - value. Useful for simulating selection to keep observable 'off'. 

    string_name: off
    """
    return 1 - value

def ff_neutral(value):
    """
    Microscopic fitness function for fx_obs. Fitness is always 1.0, modeling
    no selection on fx_obs. 

    string_name: neutral
    """
    return 1.0

# These dictionaries let us look up the fitness functions using strings. They 
# look up the name using the keyword string_name: in the function docstrings
FITNESS_STR_TO_FUNCTION = {}
FITNESS_FUNCTION_TO_STR = {}
for fcn in [ff_on,ff_off,ff_neutral]:
    string = fcn.__doc__.split("string_name:")[1].strip()
    FITNESS_STR_TO_FUNCTION[string] = fcn
    FITNESS_FUNCTION_TO_STR[fcn] = string
    

def get_fitness_function(fitness_fcn):
    """
    Get observable functions by name. 
    
    Parameters
    ----------
    fitness_fcn : str
        fitness function. should be "on", "off", "neutral"
    
    Returns
    -------
    fcn : function
        fast fitness function that takes a the observable as input
    """
    
    if fitness_fcn not in FITNESS_STR_TO_FUNCTION:
        err = f"fitness_fcn ('{fitness_fcn}') should be one of:\n"
        for k in FITNESS_STR_TO_FUNCTION:
            err += f"    {k}\n"
        err += "\n"
        raise ValueError(err)
    
    return FITNESS_STR_TO_FUNCTION[fitness_fcn]

       
def fitness_function(ens,
                     mut_energy,
                     mu_dict,
                     fitness_fcns,
                     select_on="fx_obs",
                     select_on_folded=True,
                     fitness_kwargs=None,
                     T=298.15):
    """
    Calculate fitness from the ensemble given mutations (in mut_energy), 
    chemical potentials (mu_dict), and fitness function(s). 
    
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
    fitness_fcns : function or list
        fitness function(s) to apply. Should either be a single function or list 
        of functions. Functions should take value from "select_on" as their
        first argument and **fitness_kwargs as their remaining arguments. If a 
        list, the list must be the same length as the number of conditions in 
        mu_dict. 
    select_on : str, default="fx_obs"
        observable to pass to fitness_fcns. Should be either fx_obs or dG_obs
    fitness_kwargs : dict, optional
        pass these keyword arguments to the fitness_fcn
    select_on_folded : bool, default=True
        In addition to selecting on select_on, multiply the fitness by the 
        fraction of the protein molecules that are folded. 
    T : float, default=298.15
        temperature in Kelvin. This can be an array; if so, its length must
        match the length of the arrays specified in mu_dict. 

    Returns
    -------
    df : pandas.DataFrame
        pandas dataframe holding information about the ensemble versus mu_dict,
        including fitness. 
    """

    ens = check_ensemble(ens)
    mut_energy = check_mut_energy(mut_energy)
    mu_dict, num_conditions = check_mu_dict(mu_dict)
    fitness_fcns = check_fitness_fcns(fitness_fcns,
                                      num_conditions=num_conditions)
    
    if select_on not in ["fx_obs","dG_obs"]:
        err = "select_on should be either fx_obs or dG_obs\n"
        raise ValueError(err)

    fitness_kwargs = check_fitness_kwargs(fitness_kwargs,
                                          fitness_fcns=fitness_fcns)
    select_on_folded = check_bool(value=select_on_folded,
                                  variable_name="select_on_folded")
    
    T = check_T(T=T,num_conditions=num_conditions)
        
    df = ens.get_obs(mut_energy=mut_energy,
                     mu_dict=mu_dict,
                     T=T)
    
    values = np.array(df[select_on])
    all_F = np.zeros(num_conditions)
    for i in range(num_conditions):
        all_F[i] = fitness_fcns[i](values[i],**fitness_kwargs)

    if select_on_folded:
        all_F = all_F*np.array(df["fx_folded"])

    return all_F


class FitnessContainer:
    """
    Class used to calculate fitness of genotypes in evolutionary simulations.
    It holds fixed aspects of the fitness (ensemble, chemical potentials,
    fitness functions, and temperature), allowing us to quickly calculate
    fitness given only the energy of a particular gentoype. 

    Notes
    -----
    This class stores a private *copy* of ens. This is so the ensemble z-matrix 
    stays identical for all calculations, even if the user uses the initial 
    ensemble object for a calculation with a different mu_dict after initialization
    of the FitnessContainer object. (That calculation triggers creation of new
    z-matrix, which would potentially change output observables). 
    """
    def __init__(self,
                 ens,
                 mu_dict,
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
        mu_dict : dict, optional
            dictionary of chemical potentials. keys are the names of chemical
            potentials. Values are floats or arrays of floats. Any arrays 
            specified must have the same length. If a chemical potential is not
            specified in the dictionary, its value is set to 0. 
        fitness_fcns : function or list
            fitness function(s) to apply. Should either be a single function or list 
            of functions. Functions should take value from "select_on" as their
            first argument and **fitness_kwargs as their remaining arguments. If a 
            list, the list must be the same length as the number of conditions in 
            mu_dict. 
        select_on : str, default="fx_obs"
            observable to pass to fitness_fcns. Should be either fx_obs or dG_obs
        fitness_kwargs : dict, optional
            pass these keyword arguments to the fitness_fcn
        select_on_folded : bool, default=True
            In addition to selecting on select_on, multiply the fitness by the 
            fraction of the protein molecules that are folded. 
        T : float, default=298.15
            temperature in Kelvin. This can be an array; if so, its length must
            match the length of the arrays specified in mu_dict. 
        """
        
        ens = check_ensemble(ens)
        mu_dict, num_conditions = check_mu_dict(mu_dict)
        fitness_fcns = check_fitness_fcns(fitness_fcns,
                                          num_conditions=num_conditions)
        
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
        self._mu_dict = mu_dict
        self._fitness_fcns = fitness_fcns
        self._select_on = select_on
        self._select_on_folded = select_on_folded
        self._fitness_kwargs = fitness_kwargs
        self._T = T

        self._fitness_fcns_strings = []
        for f in self._fitness_fcns:
            if f in FITNESS_FUNCTION_TO_STR:
                self._fitness_fcns_strings.append(FITNESS_FUNCTION_TO_STR[f])
            else:
                self._fitness_fcns_strings.append(f"{f}")
                
        self._private_ens.load_mu_dict(mu_dict=self._mu_dict)
        self._obs_function = self._private_ens.get_observable_function(self._select_on)
        self._num_conditions = len(self._fitness_fcns)
        self._F_array = np.zeros(self._num_conditions,dtype=float)

    def fitness(self,mut_energy_array):
        """
        Calculate the fitness of a genotype with total mutational energies 
        given by mut_energy_array. Fitness is defined as the product of the
        fitness  in each of the conditions specified in mu_dict. 

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
 
        return np.prod(self._F_array)
    
    def to_dict(self):
        """
        Return a json-able dictionary describing the fitness parameters.
        """

        to_write = ["mu_dict",
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
    def mu_dict(self):
        return self._mu_dict
    
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
