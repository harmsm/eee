
from eee.evolve.fitness.ff import ff_on
from eee.evolve.fitness.ff import ff_off
from eee.evolve.fitness.ff import ff_neutral
from eee.evolve.fitness.map_fitness_fcn_to_string import map_fitness_fcn_to_string
from eee.evolve.fitness.map_fitness_fcn_to_string import _FITNESS_FUNCTION_TO_STR
from eee.evolve.fitness.map_fitness_fcn_to_string import _FITNESS_STR_TO_FUNCTION

import pytest

def test__construct_ff_dicts():
    # Runs on initialization and creates _FITNESS_FUNCTION_TO_STR and
    # _FITNESS_STR_TO_FUNCTION. Validate those. 

    assert callable(_FITNESS_STR_TO_FUNCTION["on"])
    assert _FITNESS_STR_TO_FUNCTION["on"](1) == 1
    assert _FITNESS_STR_TO_FUNCTION["on"](0) == 0
    
    assert callable(_FITNESS_STR_TO_FUNCTION["off"])
    assert _FITNESS_STR_TO_FUNCTION["off"](1) == 0
    assert _FITNESS_STR_TO_FUNCTION["off"](0) == 1

    assert callable(_FITNESS_STR_TO_FUNCTION["neutral"])
    assert _FITNESS_STR_TO_FUNCTION["neutral"](1) == 1
    assert _FITNESS_STR_TO_FUNCTION["neutral"](0) == 1

    # Make sure mapping both directions works
    for k in _FITNESS_STR_TO_FUNCTION:
        fcn = _FITNESS_STR_TO_FUNCTION[k]
        assert _FITNESS_FUNCTION_TO_STR[fcn] == k


def test_map_fitness_fcn_to_string(variable_types):
    
    assert map_fitness_fcn_to_string(value="on",return_as="string") == "on"
    assert map_fitness_fcn_to_string(value="on",return_as="function") == ff_on
    assert map_fitness_fcn_to_string(value=ff_on,return_as="string") == "on"
    assert map_fitness_fcn_to_string(value=ff_on,return_as="function") == ff_on

    assert map_fitness_fcn_to_string(value="off",return_as="string") == "off"
    assert map_fitness_fcn_to_string(value="off",return_as="function") == ff_off
    assert map_fitness_fcn_to_string(value=ff_off,return_as="string") == "off"
    assert map_fitness_fcn_to_string(value=ff_off,return_as="function") == ff_off

    assert map_fitness_fcn_to_string(value="neutral",return_as="string") == "neutral"
    assert map_fitness_fcn_to_string(value="neutral",return_as="function") == ff_neutral
    assert map_fitness_fcn_to_string(value=ff_neutral,return_as="string") == "neutral"
    assert map_fitness_fcn_to_string(value=ff_neutral,return_as="function") == ff_neutral

    for v in variable_types["not_str"]:
        if callable(v):
            continue
        
        if issubclass(type(v),type):
            continue

        print(v,type(v),flush=True)
        with pytest.raises(ValueError):
            map_fitness_fcn_to_string(v,return_as="string")    

    for v in variable_types["everything"]:
        if callable(v):
            continue

        if issubclass(type(v),type):
            continue

        print(v,type(v),flush=True)
        with pytest.raises(ValueError):
            map_fitness_fcn_to_string(v,return_as="function")    

    for v in variable_types["everything"]:
        if callable(v):
            continue

        if issubclass(type(v),type):
            continue

        print(v,type(v),flush=True)
        with pytest.raises(ValueError):
            map_fitness_fcn_to_string("on",return_as=v)    
