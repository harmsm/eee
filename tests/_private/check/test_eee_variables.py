import pytest

import pandas as pd

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

def test_check_mu_stoich(variable_types):

    # mu_dict argument type checking
    print("--- mu_dict ---")
    for v in variable_types["dict"]:
        print(v,type(v))
        check_mu_stoich(v)
        

    for v in variable_types["not_dict"]:
        print(v,type(v))
        with pytest.raises(ValueError):
            check_mu_stoich(v)


def test_check_mu_dict(variable_types):

    for v in variable_types["dict"]:
        print(v,type(v))
        check_mu_dict(v)
        
    for v in variable_types["not_dict"]:
        print(v,type(v))
        with pytest.raises(ValueError):
            check_mu_dict(v)

    allowed = variable_types["float_value_or_iter"][:]
    not_allowed = variable_types["not_float_value_or_iter"][:]
    for m in [[],()]:
        allowed.remove(m)
        not_allowed.append(m)
    not_allowed.append(pd.DataFrame({"X":[1,2,3]}))

    for v in allowed:

        if issubclass(type(v),pd.DataFrame):
            continue

        print(v,type(v))
        mu_dict = {"X":v}
        check_mu_dict(mu_dict)

    for v in not_allowed:
        print(v,type(v))
    
        mu_dict = {"X":v}
        with pytest.raises(ValueError):
            check_mu_dict(mu_dict)

def test_check_mut_energy(variable_types):

    for v in [{},{"test1":1},{"test2":1},{"test1":1,"test2":1}]: 
        print(v,type(v))
        check_mut_energy(v)

    not_allowed = variable_types["not_dict"]
    for v in not_allowed:
        print(v,type(v))
        with pytest.raises(ValueError):
            check_mut_energy(v)

    for v in variable_types["floats_or_coercable"]:
        print(v,type(v))
        mut_energy = {"test1":v}
        check_mut_energy(mut_energy=mut_energy)
    

    for v in variable_types["not_floats_or_coercable"]:
        print(v,type(v))
        mut_energy = {"test1":v}
        with pytest.raises(ValueError):
            check_mut_energy(mut_energy=mut_energy)

def test_check_ddg_df(variable_types):
    
    not_allowed = variable_types["everything"]
    not_allowed = [n for n in not_allowed if not issubclass(type(n),pd.DataFrame)]
    
    for v in not_allowed:
        print(v,type(v))
        with pytest.raises(ValueError):
            check_ddg_df(v)

    df = pd.DataFrame({"test":[1,2,3]})
    check_ddg_df(df)


def test_check_fitness_fcns(variable_types):
    
    for v in variable_types["everything"]:
        print(v,type(v))

        # Skip empty iterables
        if hasattr(v,"__iter__") and not issubclass(type(v),type):
            if len(v) == 0:
                continue

        with pytest.raises(ValueError):
            check_fitness_fcns(v)

    fitness_fcns = [print,print]
    value = check_fitness_fcns(fitness_fcns)
    assert value[0] == fitness_fcns[0]
    assert value[1] == fitness_fcns[1]

    mu_dict = {"test":[1,2]}
    check_fitness_fcns(fitness_fcns=fitness_fcns,mu_dict=mu_dict)

    mu_dict = {"test":[1,2,3]}
    with pytest.raises(ValueError):
        check_fitness_fcns(fitness_fcns=fitness_fcns,mu_dict=mu_dict)

    mu_dict = {"test":[1]}
    with pytest.raises(ValueError):
        check_fitness_fcns(fitness_fcns=fitness_fcns,mu_dict=mu_dict)


def test_check_T(variable_types):
    
    allowed = variable_types["floats_or_coercable"][:]
    not_allowed = variable_types["not_floats_or_coercable"][:]
    for v in allowed:
        print(v,type(v))

        if float(v) <= 0:
            not_allowed.append(v)
            continue

        check_T(T=v)

    for v in not_allowed:
        print(v,type(v))

        with pytest.raises(ValueError):
            check_T(T=v)

def test_check_mutation_rate(variable_types):

    allowed = variable_types["floats_or_coercable"][:]
    not_allowed = variable_types["not_floats_or_coercable"][:]
    for v in allowed:
        print(v,type(v))

        if float(v) <= 0:
            not_allowed.append(v)
            continue

        check_mutation_rate(mutation_rate=v)

    for v in not_allowed:
        print(v,type(v))
        
        with pytest.raises(ValueError):
            check_mutation_rate(mutation_rate=v)

def test_check_population_size(variable_types):

    allowed = variable_types["ints_or_coercable"][:]
    not_allowed = variable_types["not_ints_or_coercable"][:]
    for v in allowed:
        print(v,type(v))

        if int(v) <= 0:
            not_allowed.append(v)
            continue

        check_population_size(population_size=v)

    for v in not_allowed:
        print(v,type(v))
        
        with pytest.raises(ValueError):
            check_population_size(population_size=v)

def test_check_num_generations(variable_types):

    allowed = variable_types["ints_or_coercable"][:]
    not_allowed = variable_types["not_ints_or_coercable"][:]
    for v in allowed:
        print(v,type(v))

        if int(v) <= 0:
            not_allowed.append(v)
            continue

        check_num_generations(num_generations=v)

    for v in not_allowed:
        print(v,type(v))
        
        with pytest.raises(ValueError):
            check_num_generations(num_generations=v)

def test_check_burn_in_generations(variable_types):

    allowed = variable_types["ints_or_coercable"][:]
    not_allowed = variable_types["not_ints_or_coercable"][:]
    for v in allowed:
        print(v,type(v))

        if int(v) <= 0:
            not_allowed.append(v)
            continue

        check_burn_in_generations(burn_in_generations=v)

    for v in not_allowed:
        print(v,type(v))
        
        with pytest.raises(ValueError):
            check_burn_in_generations(burn_in_generations=v)


def test_check_num_mutations(variable_types):

    allowed = variable_types["ints_or_coercable"][:]
    not_allowed = variable_types["not_ints_or_coercable"][:]
    for v in allowed:
        print(v,type(v))

        if int(v) <= 0:
            not_allowed.append(v)
            continue

        check_num_mutations(num_mutations=v)

    for v in not_allowed:
        print(v,type(v))
        
        with pytest.raises(ValueError):
            check_num_mutations(num_mutations=v)
    