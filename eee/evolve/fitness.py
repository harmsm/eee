"""
Functions and classes for calculating fitness from an ensemble during an
evolutionary simulation. 
"""

from eee._private.check.eee_variables import check_T
from eee._private.check.ensemble import check_ensemble
from eee._private.check.eee_variables import check_fitness_fcns
from eee._private.check.eee_variables import check_mu_dict
from eee._private.check.eee_variables import check_mut_energy

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

def _fitness_function(ens,
                      mut_energy,
                      mu_dict,
                      fitness_fcns,
                      select_on,
                      fitness_kwargs,
                      T):
    """
    Private fitness function without error checking. Should be called via the
    public fitness_function for use in the API.
    """
    
    num_conditions = len(fitness_fcns)

    state = ens.do_arg_checking
    ens.do_arg_checking = False
    values = ens.get_obs(mut_energy=mut_energy,
                         mu_dict=mu_dict,
                         T=T)
    ens.do_arg_checking = state
    
    all_F = np.zeros(num_conditions)
    for i in range(num_conditions):
        all_F[i] = fitness_fcns[i](values[select_on].iloc[i],**fitness_kwargs)

    return all_F
       

def fitness_function(ens,
                     mut_energy,
                     mu_dict,
                     fitness_fcns,
                     select_on="fx_obs",
                     fitness_kwargs={},
                     T=298.15):
    """
    Calculate fitness from the ensemble given mutations (in mut_energy), 
    chemical potentials (mu_dict), and fitness functions for each condition in 
    mu_dict (fitness_fcns). 
    
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
    fitness_fcns : list-like
        list of fitness functions to apply. There should be one fitness function
        for each condition specified in mu_dict. The first argument of each 
        function must be either fx_obs or dG_obs. Other keyword arguments can be
        specified in fitness_kwargs.
    select_on : str, default="fx_obs"
        observable to pass to fitness_fcns. Should be either fx_obs or dG_obs
    fitness_kwargs : dict, optional
        pass these keyword arguments to the fitness_fcn
    T : float, default=298.15
        temperature in Kelvin. This can be an array; if so, its length must
        match the length of the arrays specified in mu_dict. 

    Returns
    -------
    F : numpy.array
        float numpy array with one fitness per condition. 
    """

    ens = check_ensemble(ens)
    mut_energy = check_mut_energy(mut_energy)
    mu_dict = check_mu_dict(mu_dict)
    fitness_fcns = check_fitness_fcns(fitness_fcns,mu_dict=mu_dict)
    check_T(T=T)
        
    if select_on not in ["fx_obs","dG_obs"]:
        err = "select_on should be either fx_obs or dG_obs\n"
        raise ValueError(err)

    return _fitness_function(ens=ens,
                             mut_energy=mut_energy,
                             mu_dict=mu_dict,
                             fitness_fcns=fitness_fcns,
                             select_on=select_on,
                             fitness_kwargs=fitness_kwargs,
                             T=T)


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
                 fitness_kwargs={},
                 T=298.15):
        """
        See eee.evolve.fitness_function docstring for information on arguments.
        """
        
        ens = check_ensemble(ens)
        mu_dict = check_mu_dict(mu_dict)
        fitness_fcns = check_fitness_fcns(fitness_fcns,mu_dict=mu_dict)
        check_T(T=T)
    
        if select_on not in ["fx_obs","dG_obs"]:
            err = "select_on should be either fx_obs or dG_obs\n"
            raise ValueError(err)

        self._ens = ens
        self._mu_dict = mu_dict
        self._fitness_fcns = fitness_fcns
        self._select_on = select_on
        self._fitness_kwargs = fitness_kwargs
        self._T = T

    def fitness(self,mut_energy):
        """
        Calculate the fitness of a genotype with total mutational energies 
        given by mut_dict. Fitness is defined as the product of the fitness 
        in each of the conditions specified in mu_dict. 
        """
        
        F = _fitness_function(ens=self._ens,
                              mut_energy=mut_energy,
                              mu_dict=self._mu_dict,
                              fitness_fcns=self._fitness_fcns,
                              select_on=self._select_on,
                              fitness_kwargs=self._fitness_kwargs,
                              T=self._T)
 
        return np.prod(F)
    
    @property
    def ens(self):
        return self._ens

    @property
    def T(self):
        return self._T

