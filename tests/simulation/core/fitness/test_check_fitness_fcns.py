
from eee.simulation.core.fitness.ff import ff_on
from eee.simulation.core.fitness.ff import ff_off
from eee.simulation.core.fitness.ff import ff_neutral
from eee.simulation.core.fitness.check_fitness_fcns import _map_fitness_fcn_to_string
from eee.simulation.core.fitness.check_fitness_fcns import check_fitness_fcns
from eee.simulation import FF_AVAILABLE

from eee.simulation.core.fitness.ff import ff_on
from eee.simulation.core.fitness.ff import ff_off
from eee.simulation.core.fitness.ff import ff_neutral

import pytest

def test__get_ff_available():
    # Runs on initialization and creates FF_AVAILABLE. Validate that

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


def test_check_fitness_fcns(variable_types):
    
    # Make sure named fitness functions pass through properly
    values = check_fitness_fcns(["on","off","neutral"],num_conditions=3)
    assert values[0] == ff_on
    assert values[1] == ff_off
    assert values[2] == ff_neutral

    values = check_fitness_fcns([ff_on,ff_off,ff_neutral],num_conditions=3)
    assert values[0] == ff_on
    assert values[1] == ff_off
    assert values[2] == ff_neutral

    with pytest.raises(ValueError):
        values = check_fitness_fcns(["bad_value","bad_value"],num_conditions=2)

    # Other sorts of stuff getting passed in 
    for v in variable_types["everything"]:

        # Skip empty iterables
        if hasattr(v,"__iter__"):
            if issubclass(type(v),type):
                continue
        
            if len(v) ==  0:
                continue

        print(v,type(v),flush=True)
        with pytest.raises(ValueError):
            check_fitness_fcns(v,num_conditions=2)

    fitness_fcns = [print,print]
    value = check_fitness_fcns(fitness_fcns,num_conditions=2)
    assert value[0] == fitness_fcns[0]
    assert value[1] == fitness_fcns[1]

    check_fitness_fcns(fitness_fcns=fitness_fcns,num_conditions=2)

    with pytest.raises(ValueError):
        check_fitness_fcns(fitness_fcns=fitness_fcns,num_conditions=3)

    f = check_fitness_fcns(print,5)
    assert len(f) == 5
    assert f[0] is print
    assert f[4] is print

    # Check return_as argument
    fitness_fcns = [ff_on,ff_off]
    value = check_fitness_fcns(fitness_fcns=fitness_fcns,
                               num_conditions=2,
                               return_as="function")
    assert value[0] is ff_on
    assert value[1] is ff_off

    value = check_fitness_fcns(fitness_fcns=fitness_fcns,
                               num_conditions=2,
                               return_as="string")
    assert value[0] == "on"
    assert value[1] == "off"

    with pytest.raises(ValueError):
        value = check_fitness_fcns(fitness_fcns=fitness_fcns,
                            num_conditions=2,
                            return_as="bad_return_as")