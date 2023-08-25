"""
Validate a mutant energy dictionary.
"""

from eee._private.check.standard import check_float

def check_mut_energy(mut_energy):
    """
    Validate a mutant energy dictionary.
    
    Parameters
    ----------
    mut_energy : dict
        mutant energy as a dictionary, where keys are species and values are 
        energetic effects (floats)
    
    Returns
    -------
    mut_energy : dict
        validated mutant energy dictionary
    """

    # Make sure mut_energy is a dictionary of floats
    if not issubclass(type(mut_energy),dict) or issubclass(type(mut_energy),type):
        err = "mut_energy should be a dictionary that keys chemical species to effects of mutations\n"
        raise ValueError(err)
    
    for s in mut_energy:
        mut_energy[s] = check_float(value=mut_energy[s],
                                    variable_name=f"mut_energy['{s}']")
    
    return mut_energy