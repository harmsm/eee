"""
Function to validate ligand_stoich.
"""
from eee._private.check.standard import check_float

def check_ligand_stoich(ligand_stoich):
    """
    Validate the ligand_stoich dictionary.

    Parameters
    ----------
    ligand_stoich : dict
        dictionary keying chemical species to stoichiometry

    Returns
    -------
    ligand_stoich : dict
        validate ligand_stoich dictionary
    """

    if not issubclass(type(ligand_stoich),dict):
        err = "ligand_stoich should be a dictionary that keys chemical species to stoichiometry\n"
        raise ValueError(err)
    
    for lig in ligand_stoich:
        ligand_stoich[lig] = check_float(value=ligand_stoich[lig],
                                    variable_name=f"ligand_stoich['{lig}']",
                                    minimum_allowed=0,
                                    minimum_inclusive=True)
        
    return ligand_stoich