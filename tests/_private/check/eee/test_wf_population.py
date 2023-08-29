import pytest

from eee._private.check.eee import check_wf_population
from eee._private.check.standard import check_int

import numpy as np

def test_check_wf_population(variable_types):


    simple_pops = [1,[0],(0,),np.zeros(1,dtype=int),{0:1}]
    for p in simple_pops:
        population = check_wf_population(p)
        assert issubclass(type(population),np.ndarray)
        assert len(population) == 1
        assert population[0] == 0

    simple_pops = [50,
                   [0 for _ in range(50)],
                   tuple([0 for _ in range(50)]),
                   np.zeros(50,dtype=int),
                   {0:50}]
    for p in simple_pops:
        population = check_wf_population(p)
        assert issubclass(type(population),np.ndarray)
        assert len(population) == 50
        assert population[0] == 0
        assert len(np.unique(population)) == 1

    # Correct dict passing
    population = check_wf_population({0:10,1:10})
    assert issubclass(type(population),np.ndarray)
    assert len(population) == 20
    assert population[0] == 0
    assert population[10] == 1
    assert len(np.unique(population)) == 2

    # Zero or negative pop values, otherwise good
    for v in [0,[],tuple([]),np.zeros(0,dtype=int),{0:0},{},-1]:
        print(v,type(v),flush=True)
        with pytest.raises(ValueError):
            population = check_wf_population(population=v)

    for v in variable_types["everything"]:
        print(v,type(v),flush=True)
    
        # If it coerces to a positive int, move along
        try:
            check_int(v,minimum_allowed=1)
            continue
        except ValueError:
            pass

        # If it coerces to an int numpy array, move along
        try:
            np.array(v,dtype=int)
            continue
        except (TypeError,ValueError,OverflowError):
            pass

        # If an dict of ints, move along
        if issubclass(type(v),dict) and not issubclass(type(v),type):
            if len(v.keys()) > 0 and issubclass(type(list(v.keys())[0]),int):
                continue

        with pytest.raises(ValueError):
            population = check_wf_population(population=v)
        