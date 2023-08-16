import pytest

from eee._private.check.eee_variables import check_mu_stoich
from eee._private.check.eee_variables import check_mu_dict
from eee._private.check.eee_variables import check_mut_energy
from eee._private.check.eee_variables import check_ddg_df
from eee._private.check.eee_variables import check_fitness_fcns
from eee._private.check.eee_variables import check_T
from eee._private.check.eee_variables import check_mutation_rate
from eee._private.check.eee_variables import check_population_size
from eee._private.check.eee_variables import check_num_generations
from eee._private.check.eee_variables import check_burn_in_generations
from eee._private.check.eee_variables import check_num_mutations

import pandas as pd
import numpy as np

def test_check_mu_stoich(variable_types):

    # mu_dict argument type checking
    print("--- mu_dict ---")
    for v in variable_types["dict"]:
        print(v,type(v),flush=True)
        check_mu_stoich(v)
        

    for v in variable_types["not_dict"]:
        print(v,type(v),flush=True)
        with pytest.raises(ValueError):
            check_mu_stoich(v)


def test_check_mu_dict(variable_types):
        
    for v in variable_types["not_dict"]:
        print(v,type(v),flush=True)
        with pytest.raises(ValueError):
            check_mu_dict(v)

    out, length = check_mu_dict({})
    assert issubclass(type(out),dict)
    assert len(out) == 0
    assert length == 1

    bad_types = variable_types["types"][:]
    bad_types.extend([pd.DataFrame({"a":[1,2,3]}),
                      {"a":[1,2,3]}])
    for v in bad_types:
        print(v,type(v),flush=True)
        with pytest.raises(ValueError):
            check_mu_dict({"X":v})

    bad_values = [[],np.array([])]
    for v in bad_values:
        print(v,type(v),flush=True)
        with pytest.raises(ValueError):
            check_mu_dict({"X":v})

    bad_values = ["should_be_a_number"]
    for v in bad_values:
        print(v,type(v),flush=True)
        with pytest.raises(ValueError):
            check_mu_dict({"X":v})

    good_values = variable_types["floats_or_coercable"][:]
    good_values.extend([[1,2,3],
                        [1.0,2.0,3.0],
                        [1,2.0,3],
                        np.array([1,2,3],dtype=int),
                        np.array([1,2,3],dtype=float)])
    for v in good_values:
        print(v,type(v),flush=True)
        check_mu_dict({"X":v})
    
    good_values = [{"X":1,"Y":1},
                   {"X":"1","Y":"1"},
                   {"X":1,"Y":[1]},
                   {"X":[1],"Y":[1]}]
    for v in good_values:
        print(v,type(v),flush=True)
        result, length = check_mu_dict(v)
        assert length == 1
        assert np.array_equal(result["X"],[1])
        assert np.array_equal(result["Y"],[1])

    good_values = [{"X":1,"Y":[1,2,3]},
                   {"X":[1],"Y":[1,2,3]},
                   {"X":[1,1,1],"Y":[1,2,3]}]
    for v in good_values:
        print(v,type(v),flush=True)
        result, length = check_mu_dict(v)
        assert length == 3
        assert np.array_equal(result["X"],[1,1,1])
        assert np.array_equal(result["Y"],[1,2,3])

    good_values = [{"X":1,"Y":[1,2,3],"Z":[2]},
                   {"X":[1],"Y":[1,2,3],"Z":[2,2,2]},
                   {"X":[1,1,1],"Y":[1,2,3],"Z":[2,2,2]}]
    for v in good_values:
        print(v,type(v),flush=True)
        result, length = check_mu_dict(v)
        assert length == 3
        assert np.array_equal(result["X"],[1,1,1])
        assert np.array_equal(result["Y"],[1,2,3])
        assert np.array_equal(result["Z"],[2,2,2])

    bad_values = [{"X":[1,2],"Y":[1,2,3]},
                  {"X":[1,2],"Y":[1,2,3],"Z":2}]
    for v in bad_values:
        print(v,type(v),flush=True)
        with pytest.raises(ValueError):
            check_mu_dict(v)

def test_check_mut_energy(variable_types):

    for v in [{},{"test1":1},{"test2":1},{"test1":1,"test2":1}]: 
        print(v,type(v),flush=True)
        check_mut_energy(v)

    not_allowed = variable_types["not_dict"]
    for v in not_allowed:
        print(v,type(v),flush=True)
        with pytest.raises(ValueError):
            check_mut_energy(v)

    for v in variable_types["floats_or_coercable"]:
        print(v,type(v),flush=True)
        mut_energy = {"test1":v}
        check_mut_energy(mut_energy=mut_energy)
    

    for v in variable_types["not_floats_or_coercable"]:
        print(v,type(v),flush=True)
        mut_energy = {"test1":v}
        with pytest.raises(ValueError):
            check_mut_energy(mut_energy=mut_energy)

def test_check_ddg_df(variable_types):
    
    not_allowed = variable_types["everything"]
    not_allowed = [n for n in not_allowed if not issubclass(type(n),pd.DataFrame)]
    
    for v in not_allowed:
        print(v,type(v),flush=True)
        with pytest.raises(ValueError):
            check_ddg_df(v)

    df = pd.DataFrame({"test":[1,2,3]})
    check_ddg_df(df)


def test_check_fitness_fcns(variable_types):
    
    for v in variable_types["everything"]:

        # Skip empty iterables
        if hasattr(v,"__iter__") and not issubclass(type(v),type):
            if len(v) == 0:
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


def test_check_T(variable_types):
    
    allowed = variable_types["floats_or_coercable"][:]
    not_allowed = variable_types["not_floats_or_coercable"][:]
    for v in allowed:
        print(v,type(v),flush=True)

        if float(v) <= 0:
            not_allowed.append(v)
            continue

        out = check_T(T=v,num_conditions=1)
        assert len(out) == 1
        assert out[0] == float(v)

    for v in not_allowed:
        print(v,type(v),flush=True)

        with pytest.raises(ValueError):
            check_T(T=v,num_conditions=1)
    
    allowed = [[1,2],(1,2),np.array([1,2])]
    not_allowed = [["a","b"]]
    for v in allowed:
        out = check_T(T=v,num_conditions=2)
        assert len(out) == 2
        assert out[0] == float(v[0])
        assert out[1] == float(v[1])

        with pytest.raises(ValueError):
            out = check_T(T=v,num_conditions=3)

def test_check_mutation_rate(variable_types):

    allowed = variable_types["floats_or_coercable"][:]
    not_allowed = variable_types["not_floats_or_coercable"][:]
    for v in allowed:
        print(v,type(v),flush=True)

        if float(v) <= 0:
            not_allowed.append(v)
            continue

        check_mutation_rate(mutation_rate=v)

    for v in not_allowed:
        print(v,type(v),flush=True)
        
        with pytest.raises(ValueError):
            check_mutation_rate(mutation_rate=v)

def test_check_population_size(variable_types):

    allowed = variable_types["ints_or_coercable"][:]
    not_allowed = variable_types["not_ints_or_coercable"][:]
    for v in allowed:
        print(v,type(v),flush=True)

        if int(v) <= 0:
            not_allowed.append(v)
            continue

        check_population_size(population_size=v)

    for v in not_allowed:
        print(v,type(v),flush=True)
        
        with pytest.raises(ValueError):
            check_population_size(population_size=v)

def test_check_num_generations(variable_types):

    allowed = variable_types["ints_or_coercable"][:]
    not_allowed = variable_types["not_ints_or_coercable"][:]
    for v in allowed:
        print(v,type(v),flush=True)

        if int(v) <= 0:
            not_allowed.append(v)
            continue

        check_num_generations(num_generations=v)

    for v in not_allowed:
        print(v,type(v),flush=True)
        
        with pytest.raises(ValueError):
            check_num_generations(num_generations=v)

def test_check_burn_in_generations(variable_types):

    allowed = variable_types["ints_or_coercable"][:]
    not_allowed = variable_types["not_ints_or_coercable"][:]
    for v in allowed:
        print(v,type(v),flush=True)

        if int(v) <= 0:
            not_allowed.append(v)
            continue

        check_burn_in_generations(burn_in_generations=v)

    for v in not_allowed:
        print(v,type(v),flush=True)
        
        with pytest.raises(ValueError):
            check_burn_in_generations(burn_in_generations=v)


def test_check_num_mutations(variable_types):

    allowed = variable_types["ints_or_coercable"][:]
    not_allowed = variable_types["not_ints_or_coercable"][:]
    for v in allowed:
        print(v,type(v),flush=True)

        if int(v) <= 0:
            not_allowed.append(v)
            continue

        check_num_mutations(num_mutations=v)

    for v in not_allowed:
        print(v,type(v),flush=True)
        
        with pytest.raises(ValueError):
            check_num_mutations(num_mutations=v)
    