from .analysis import get_genotype_frequencies
from .analysis import extract_alignment

from .fitness import FitnessContainer
from .fitness import fitness_function
from .fitness import ff_list

from .genotype import GenotypeContainer

from .base import container

from . import calculation

def _get_allowable_calcs():
    """
    Look in the calculation module for calculations that can be used in json
    files. 
    """

    import inspect

    allowable_calcs = {}

    possibilities = inspect.getmembers(calculation)
    for p in possibilities:

        is_good = False
        try:
            is_good = issubclass(p[1],container.SimulationContainer)
        except TypeError:
            continue
            
        if is_good:
            allowable_calcs[p[1].calc_type] = p[1]

    return allowable_calcs

_ALLOWABLE_CALCS = _get_allowable_calcs()
calc_list = list(_ALLOWABLE_CALCS.keys())
calc_list.sort()