"""
Function for calculating fitness from an ensemble.
"""

from eee._private.check.eee_variables import check_T
from eee._private.check.ensemble import check_ensemble
from eee._private.check.eee_variables import check_fitness_fcns
from eee._private.check.eee_variables import check_fitness_kwargs
from eee._private.check.eee_variables import check_mu_dict
from eee._private.check.eee_variables import check_mut_energy
from eee._private.check.standard import check_bool

import numpy as np

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
