"""
Code for running evolutionary simulations on thermodynamic ensembles. 
"""
from . import analysis
from . import calcs
from . import core

import inspect

def _get_calc_available():
    """
    Look for calcs in submodule. This creates the CALC_AVAILABLE dict which 
    can be used to write/parse json. 
    """

    available_calcs = {}

    possibilities = inspect.getmembers(calcs)
    for p in possibilities:

        is_good = False
        try:
            is_good = issubclass(p[1],core.simulation.Simulation)
        except TypeError:
            continue
            
        if is_good:
            available_calcs[p[1].calc_type] = p[1]

    return available_calcs

from .core import FF_AVAILABLE
CALC_AVAILABLE = _get_calc_available()


# This has to go down here because io.read_json depends on CALC_AVAILABLE, 
# which we define above. 
from . import io
