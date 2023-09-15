
from eee.simulation.core.fitness.ff import ff_on
from eee.simulation.core.fitness.ff import ff_off
from eee.simulation.core.fitness.ff import ff_neutral
from eee.simulation.core.fitness.ff import ff_on_above
from eee.simulation.core.fitness.ff import ff_on_below

from eee.simulation.core.fitness.map_fitness_fcn import map_fitness_fcn
from eee.simulation import FF_AVAILABLE


import pytest

def test__get_ff_available():
    
    assert callable(FF_AVAILABLE["on"])
    assert FF_AVAILABLE["on"](1) == 1
    assert FF_AVAILABLE["on"](0) == 0
    
    assert callable(FF_AVAILABLE["off"])
    assert FF_AVAILABLE["off"](1) == 0
    assert FF_AVAILABLE["off"](0) == 1

    assert callable(FF_AVAILABLE["neutral"])
    assert FF_AVAILABLE["neutral"](1) == 1
    assert FF_AVAILABLE["neutral"](0) == 1

    assert callable(FF_AVAILABLE["on_above"])
    assert FF_AVAILABLE["on_above"](0.55,threshold=0.5) == 1
    assert FF_AVAILABLE["on_above"](0.45,threshold=0.5) == 0

    assert callable(FF_AVAILABLE["on_below"])
    assert FF_AVAILABLE["on_below"](0.55,threshold=0.5) == 0
    assert FF_AVAILABLE["on_below"](0.45,threshold=0.5) == 1


def test_map_fitness_fcn(variable_types):
    
    assert map_fitness_fcn(value="on",return_as="string") == "on"
    assert map_fitness_fcn(value="on",return_as="function") == ff_on
    assert map_fitness_fcn(value=ff_on,return_as="string") == "on"
    assert map_fitness_fcn(value=ff_on,return_as="function") == ff_on

    assert map_fitness_fcn(value="off",return_as="string") == "off"
    assert map_fitness_fcn(value="off",return_as="function") == ff_off
    assert map_fitness_fcn(value=ff_off,return_as="string") == "off"
    assert map_fitness_fcn(value=ff_off,return_as="function") == ff_off

    assert map_fitness_fcn(value="neutral",return_as="string") == "neutral"
    assert map_fitness_fcn(value="neutral",return_as="function") == ff_neutral
    assert map_fitness_fcn(value=ff_neutral,return_as="string") == "neutral"
    assert map_fitness_fcn(value=ff_neutral,return_as="function") == ff_neutral

    assert map_fitness_fcn(value="on_above",return_as="string") == "on_above"
    assert map_fitness_fcn(value="on_above",return_as="function") == ff_on_above
    assert map_fitness_fcn(value=ff_on_above,return_as="string") == "on_above"
    assert map_fitness_fcn(value=ff_on_above,return_as="function") == ff_on_above

    assert map_fitness_fcn(value="on_below",return_as="string") == "on_below"
    assert map_fitness_fcn(value="on_below",return_as="function") == ff_on_below
    assert map_fitness_fcn(value=ff_on_below,return_as="string") == "on_below"
    assert map_fitness_fcn(value=ff_on_below,return_as="function") == ff_on_below

    for v in variable_types["not_str"]:
        if callable(v):
            continue
        
        if issubclass(type(v),type):
            continue

        print(v,type(v),flush=True)
        with pytest.raises(ValueError):
            map_fitness_fcn(v,return_as="string")    

    for v in variable_types["everything"]:
        if callable(v):
            continue

        if issubclass(type(v),type):
            continue

        print(v,type(v),flush=True)
        with pytest.raises(ValueError):
            map_fitness_fcn(v,return_as="function")    

    for v in variable_types["everything"]:
        if callable(v):
            continue

        if issubclass(type(v),type):
            continue

        print(v,type(v),flush=True)
        with pytest.raises(ValueError):
            map_fitness_fcn("on",return_as=v)    
