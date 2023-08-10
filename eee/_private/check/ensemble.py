
from eee.ensemble import Ensemble

def check_ensemble(ens,check_obs=False):

    if not issubclass(type(ens),Ensemble):
        err = "ens should be an instance of the Ensemble class"
        raise ValueError(f"\n{err}\n\n")
    
    # This will throw an error if the ensemble is improperly initialized 
    # for an observable calculation (too few species, no observable species,
    # etc.)
    if check_obs:
        ens.get_obs()

    return ens