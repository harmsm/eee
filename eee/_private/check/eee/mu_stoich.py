"""
Function to validate mu_stoich.
"""
from eee._private.check.standard import check_float

def check_mu_stoich(mu_stoich):
    """
    Validate the mu_stoich dictionary.

    Parameters
    ----------
    mu_stoich : dict
        dictionary keying chemical species to stoichiometry

    Returns
    -------
    mu_stoich : dict
        validate mu_stoich dictionary
    """

    if not issubclass(type(mu_stoich),dict):
        err = "mu_stoich should be a dictionary that keys chemical species to stoichiometry\n"
        raise ValueError(err)
    
    for mu in mu_stoich:
        mu_stoich[mu] = check_float(value=mu_stoich[mu],
                                    variable_name=f"mu_stoich['{mu}']",
                                    minimum_allowed=0,
                                    minimum_inclusive=True)
        
    return mu_stoich