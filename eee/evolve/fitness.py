
"""
"""

import numpy as np

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

    values = ens.get_obs(mut_energy=mut_energy,
                         mu_dict=mu_dict,
                         T=T)
    
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

    num_conditions = len(mu_dict[list(mu_dict.keys())[0]])

    if len(fitness_fcns) != num_conditions:
        err = "fitness should be the same length as the number of conditions\n"
        err += "in mu_dict.\n"
        raise ValueError(err)

    for f in fitness_fcns:
        if not callable(f):
            err = "Elements of the fitness vector must all be functions that\n"
            err += "take the values specified in `select_on` as inputs and\n"
            err += "return the absolute fitness.\n"
            raise ValueError(err)
        
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
    """

    def __init__(self,
                 ens,
                 mu_dict,
                 fitness_fcns,
                 select_on="fx_obs",
                 fitness_kwargs={},
                 T=298.15):
        """
        """
        
        self._ens = ens
        self._mu_dict = mu_dict
        self._fitness_fcns = fitness_fcns
        self._select_on = select_on
        self._fitness_kwargs = fitness_kwargs
        self._T = T

    def fitness(self,mut_energy):
        """
        """
        
        F = _fitness_function(ens=self._ens,
                              mut_energy=mut_energy,
                              mu_dict=self._mu_dict,
                              fitness_fcns=self._fitness_fcns,
                              select_on=self._select_on,
                              fitness_kwargs=self._fitness_kwargs,
                              T=self._T)

        return np.prod(F)