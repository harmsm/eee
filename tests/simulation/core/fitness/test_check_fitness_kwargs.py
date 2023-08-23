import pytest

from eee.simulation.core.fitness.check_fitness_kwargs import check_fitness_kwargs


def test_check_fitness_kwargs(variable_types):

    for v in variable_types["everything"]:
        if v is None:
            continue
        if issubclass(type(v),dict):
            continue
        print(v,type(v),flush=True)
        with pytest.raises(ValueError):
            check_fitness_kwargs(v)
    
    out = check_fitness_kwargs(None)
    assert issubclass(type(out),dict)
    assert len(out) == 0
    out = check_fitness_kwargs({})
    assert issubclass(type(out),dict)
    assert len(out) == 0

    out = check_fitness_kwargs(fitness_kwargs={"test":1})
    assert len(out) == 1
    assert out["test"] == 1

    with pytest.raises(ValueError):
        check_fitness_kwargs({1:"test"})
    
    def test_fcn(value):
        pass

    out = check_fitness_kwargs({},fitness_fcns=[test_fcn])
    assert len(out) == 0

    with pytest.raises(ValueError):
        out = check_fitness_kwargs({"arg":1},fitness_fcns=[test_fcn])

    def test_fcn(value,arg=1):
        pass

    out = check_fitness_kwargs({},fitness_fcns=[test_fcn])
    assert len(out) == 0

    out = check_fitness_kwargs({"arg":1},fitness_fcns=[test_fcn])
    assert out["arg"] == 1

    with pytest.raises(ValueError):
        out = check_fitness_kwargs({"arg":1,"arg2":1},fitness_fcns=[test_fcn])

    out = check_fitness_kwargs({"arg":1},fitness_fcns=[])
    assert out["arg"] == 1
