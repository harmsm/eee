"""
Functions and classes for calculating fitness from an ensemble during an
evolutionary simulation. 
"""

from eee._private.check.eee_variables import check_T
from eee._private.check.ensemble import check_ensemble
from eee._private.check.eee_variables import check_fitness_fcns
from eee._private.check.eee_variables import check_mu_dict
from eee._private.check.eee_variables import check_mut_energy
from eee._private.check.standard import check_bool

import numpy as np

def ff_on(value):
    """
    Microscopic fitness function for fx_obs. Fitness is linearly proportional
    to value. Useful for simulating selection to keep observable 'on'.
    """
    return value

def ff_off(value):
    """
    Microscopic fitness function for fx_obs. Fitness is linearly proportional
    to 1 - value. Useful for simulating selection to keep observable 'off'. 
    """
    return 1 - value

def ff_neutral(value):
    """
    Microscopic fitness function for fx_obs. Fitness is always 1.0, modeling
    no selection on fx_obs. 
    """
    return 1.0

       
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

    T = check_T(T=T,
                num_conditions=num_conditions)
        
    if select_on not in ["fx_obs","dG_obs"]:
        err = "select_on should be either fx_obs or dG_obs\n"
        raise ValueError(err)

    if fitness_kwargs is None:
        fitness_kwargs = {}

    select_on_folded = check_bool(value=select_on_folded,
                                  variable_name="select_on_folded")

    num_conditions = len(fitness_fcns)

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
    Convenience class used in evolutionary simulations. Holds onto fixed 
    aspects of the fitness (ensemble, chemical potentials, fitness functions,
    and temperature), allowing us to calculate fitness given only the 
    mut_energy of a particular gentoype. 
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
        T = check_T(T=T,
                    num_conditions=num_conditions)
    
        obs_functions = {"fx_obs":ens.get_fx_obs_fast,
                         "dG_obs":ens.get_dG_obs_fast}
        if select_on not in obs_functions:
            err = "select_on should be one of:\n"
            for k in obs_functions:
                err += f"    {k}\n"
            err += "\n"
            raise ValueError(err)
        select_on_folded = check_bool(value=select_on_folded,
                                     variable_name="select_on_folded")

        if fitness_kwargs is None:
            fitness_kwargs = {}

        self._ens = ens
        self._mu_dict = mu_dict
        self._fitness_fcns = fitness_fcns
        self._select_on = select_on
        self._select_on_folded = select_on_folded
        self._fitness_kwargs = fitness_kwargs
        self._T = T

        self._ens.load_mu_dict(mu_dict=self._mu_dict)
        self._obs_function = obs_functions[self._select_on]
        self._num_conditions = len(self._fitness_fcns)
        self._F_array = np.zeros(self._num_conditions,dtype=float)

    def fitness(self,mut_energy_array):
        """
        Calculate the fitness of a genotype with total mutational energies 
        given by mut_dict. Fitness is defined as the product of the fitness 
        in each of the conditions specified in mu_dict. 

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
    
    @property
    def ens(self):
        return self._ens

    @property
    def T(self):
        return self._T

