"""
Function for calculating fitness from an ensemble.
"""

from eee.simulation.core import Fitness

def ensemble_fitness(ens,
                     mut_energy,
                     ligand_dict,
                     fitness_fcns,
                     select_on="fx_obs",
                     select_on_folded=True,
                     fitness_kwargs=None,
                     T=298.15):
    """
    Calculate fitness from the ensemble given mutations (in mut_energy), 
    chemical potentials (ligand_dict), and fitness function(s). 
    
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
    ligand_dict : dict, optional
        dictionary of chemical potentials. keys are the names of ligands. Values
        are floats or arrays of floats of ligand chemical potential. Any arrays 
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

    Returns
    -------
    df : pandas.DataFrame
        pandas dataframe holding information about the ensemble versus ligand_dict,
        including fitness. 
    """


    fc = Fitness(ens=ens,
                 ligand_dict=ligand_dict,
                 fitness_fcns=fitness_fcns,
                 select_on=select_on,
                 select_on_folded=select_on_folded,
                 fitness_kwargs=fitness_kwargs,
                 T=T)
        
    df = ens.get_obs(mut_energy=mut_energy,
                     ligand_dict=ligand_dict,
                     T=T)

    mut_energy_array = ens.mut_dict_to_array(mut_energy=mut_energy)
    fitness_values = fc.fitness(mut_energy_array=mut_energy_array)
    df["fitness"] = fitness_values
    
    return df
