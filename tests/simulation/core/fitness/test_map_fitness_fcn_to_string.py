
from eee.simulation.core.fitness.ff import ff_on
from eee.simulation.core.fitness.ff import ff_off
from eee.simulation.core.fitness.ff import ff_neutral
from eee.simulation.core.fitness.check_fitness_fcns import _map_fitness_fcn_to_string
from eee.simulation import FF_AVAILABLE

import pytest

def test__construct_ff_dicts():
    # Runs on initialization and creates _FITNESS_FUNCTION_TO_STR and
    # _FITNESS_STR_TO_FUNCTION. Validate those. 

    assert callable(FF_AVAILABLE["on"])
    assert FF_AVAILABLE["on"](1) == 1
    assert FF_AVAILABLE["on"](0) == 0
    
    assert callable(FF_AVAILABLE["off"])
    assert FF_AVAILABLE["off"](1) == 0
    assert FF_AVAILABLE["off"](0) == 1

    assert callable(FF_AVAILABLE["neutral"])
    assert FF_AVAILABLE["neutral"](1) == 1
    assert FF_AVAILABLE["neutral"](0) == 1


def test__map_fitness_fcn_to_string(variable_types):
    
    assert _map_fitness_fcn_to_string(value="on",return_as="string") == "on"
    assert _map_fitness_fcn_to_string(value="on",return_as="function") == ff_on
    assert _map_fitness_fcn_to_string(value=ff_on,return_as="string") == "on"
    assert _map_fitness_fcn_to_string(value=ff_on,return_as="function") == ff_on

    assert _map_fitness_fcn_to_string(value="off",return_as="string") == "off"
    assert _map_fitness_fcn_to_string(value="off",return_as="function") == ff_off
    assert _map_fitness_fcn_to_string(value=ff_off,return_as="string") == "off"
    assert _map_fitness_fcn_to_string(value=ff_off,return_as="function") == ff_off

    assert _map_fitness_fcn_to_string(value="neutral",return_as="string") == "neutral"
    assert _map_fitness_fcn_to_string(value="neutral",return_as="function") == ff_neutral
    assert _map_fitness_fcn_to_string(value=ff_neutral,return_as="string") == "neutral"
    assert _map_fitness_fcn_to_string(value=ff_neutral,return_as="function") == ff_neutral

    for v in variable_types["not_str"]:
        if callable(v):
            continue
        
        if issubclass(type(v),type):
            continue

        print(v,type(v),flush=True)
        with pytest.raises(ValueError):
            _map_fitness_fcn_to_string(v,return_as="string")    

    for v in variable_types["everything"]:
        if callable(v):
            continue

        if issubclass(type(v),type):
            continue

        print(v,type(v),flush=True)
        with pytest.raises(ValueError):
            _map_fitness_fcn_to_string(v,return_as="function")    

    for v in variable_types["everything"]:
        if callable(v):
            continue

        if issubclass(type(v),type):
            continue

        print(v,type(v),flush=True)
        with pytest.raises(ValueError):
            _map_fitness_fcn_to_string("on",return_as=v)    
