import pytest

import pandas as pd

from eee._private.check.eee_variables import check_mu_stoich
from eee._private.check.eee_variables import check_mu_dict
from eee._private.check.eee_variables import check_mut_energy
from eee._private.check.eee_variables import check_ddg_df
from eee._private.check.eee_variables import check_fitness_fcns
from eee._private.check.eee_variables import check_calc_params

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

def test_check_ddg_df():
    pass

def test_check_fitness_fcns():
    pass

def test_check_calc_params(variable_types):
    pass
