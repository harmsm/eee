import pytest

from eee.ensemble import _array_expander
from eee.ensemble import Ensemble

import numpy as np
import pandas as pd

def test__array_expander():

    out, length = _array_expander({})
    assert issubclass(type(out),dict)
    assert length == 0

    out, length = _array_expander([])
    assert issubclass(type(out),list)
    assert length == 0

    # Multiple lengths
    with pytest.raises(ValueError):
        _array_expander([[1,2],[1]])

    # Multiple lengths
    with pytest.raises(ValueError):
        _array_expander({"test":[1,2],"this":[1]})

    out, length = _array_expander([[1,2],1])
    assert issubclass(type(out),list)
    assert length == 2
    assert np.array_equal(out[0],[1,2])
    assert np.array_equal(out[1],[1,1])

    out, length = _array_expander({"test":[1,2],"this":1})
    assert issubclass(type(out),dict)
    assert length == 2
    assert np.array_equal(out["test"],[1,2])
    assert np.array_equal(out["this"],[1,1])
    
    with pytest.raises(TypeError):
        out, length = _array_expander(1)
    
    out, length = _array_expander([1,2,3])
    assert np.array_equal(out,[1,2,3])
    assert length == 0


def test_Ensemble():

    ens = Ensemble()
    assert issubclass(type(ens),Ensemble)
    
    ens = Ensemble(R=1)
    assert ens._R == 1


def test_Ensemble_add_species(variable_types):
    
    ens = Ensemble()
    ens.add_species(name="test",
                    observable=False,
                    dG0=-1,
                    mu_stoich=None)
    
    assert ens._species["test"]["observable"] == False
    assert ens._species["test"]["dG0"] == -1
    assert ens._species["test"]["mu_stoich"] == {}
    assert np.array_equal(ens._mu_list,[])

    # Check for something already in ensemble
    with pytest.raises(ValueError):
        ens.add_species(name="test",
                observable=False,
                dG0=-1,
                mu_stoich=None)
        
    ens.add_species(name="another",
                    observable=True,
                    mu_stoich={"X":1})
    
    assert ens._species["another"]["observable"] == True
    assert ens._species["another"]["dG0"] == 0
    assert ens._species["another"]["mu_stoich"]["X"] == 1
    assert np.array_equal(ens._mu_list,["X"])

    # observable argument type checking
    print("--- observable ---")
    for v in variable_types["bools"]:
        print(v,type(v))
        ens = Ensemble()
        ens.add_species(name="test",observable=v)
        assert ens._species["test"]["observable"] == bool(v)

    for v in variable_types["not_bools"]:
        print(v,type(v))
        with pytest.raises(ValueError):
            ens.add_species(name="test",observable=v)

    # dG0 argument type checking
    print("--- dG0 ---")
    for v in variable_types["floats_or_coercable"]:
        print(v,type(v))
        ens = Ensemble()
        ens.add_species(name="test",dG0=v)
        assert ens._species["test"]["dG0"] == float(v)

    for v in variable_types["not_floats_or_coercable"]:
        print(v,type(v))
        ens = Ensemble()
        with pytest.raises(ValueError):
            ens.add_species(name="test",dG0=v)

    # mu_dict argument type checking
    print("--- mu_dict ---")
    for v in variable_types["dict"]:
        print(v,type(v))
        ens = Ensemble()
        ens.add_species(name="test",mu_stoich=v)

    for v in variable_types["not_dict"]:
        print(v,type(v))

        # None is allowed
        if v is None:
            continue

        ens = Ensemble()
        with pytest.raises(ValueError):
            ens.add_species(name="test",mu_stoich=v)


def test_Ensemble_get_species_dG(variable_types):

    ens = Ensemble()
    ens.add_species(name="test",
                    observable=False,
                    dG0=-1,
                    mu_stoich=None)

    with pytest.raises(ValueError):
        ens.get_species_dG("not_a_species")
    
    assert ens.get_species_dG("test") == -1
    assert ens.get_species_dG("test",mut_energy=10) == 9
    assert ens.get_species_dG("test",mut_energy=10,mu_dict={"X":10}) == 9

    ens.add_species(name="another",
                    observable=True,
                    mu_stoich={"X":1})
    
    assert ens.get_species_dG("another") == 0
    assert ens.get_species_dG("another",mut_energy=10) == 10
    assert ens.get_species_dG("another",mut_energy=10,mu_dict={"X":10}) == 20

    # Pass in an array of mu X
    value = ens.get_species_dG("another",mut_energy=10,mu_dict={"X":np.arange(10)})
    assert np.array_equal(value,np.arange(10)+10)

    # Stoichiometry of 2
    ens = Ensemble()
    ens.add_species(name="stoich2",
                    observable=False,
                    dG0=-5,
                    mu_stoich={"X":2})
    value = ens.get_species_dG("stoich2",mut_energy=10,mu_dict={"X":np.arange(10)})
    assert np.array_equal(value,np.arange(10)*2 + 5)

    # Two chemical potentials same species
    ens = Ensemble()
    ens.add_species(name="two_mu",
                    observable=False,
                    dG0=-5,
                    mu_stoich={"X":2,"Y":1})
    value = ens.get_species_dG("two_mu",
                               mut_energy=0,
                               mu_dict={"X":np.arange(10),
                                        "Y":np.arange(10)})
    assert np.array_equal(value,np.arange(10)*2+np.arange(10)-5)

    # mut_energy argument type checking
    print("--- mut_energy ---")
    for v in variable_types["floats_or_coercable"]:
        print(v,type(v))
        ens = Ensemble()
        ens.add_species(name="test")
        assert ens.get_species_dG(name="test",mut_energy=v) == float(v)

    for v in variable_types["not_floats_or_coercable"]:
        print(v,type(v))
        ens = Ensemble()
        ens.add_species(name="test")
        with pytest.raises(ValueError):
            ens.get_species_dG(name="test",mut_energy=v)

    # mu_dict argument type checking
    print("--- mu_dict ---")
    for v in variable_types["dict"]:
        print(v,type(v))
        ens = Ensemble()
        ens.add_species(name="test")
        ens.get_species_dG(name="test",mu_dict=v)

    for v in variable_types["not_dict"]:
        print(v,type(v))

        # none okay
        if v is None:
            continue

        ens = Ensemble()
        ens.add_species(name="test")
        with pytest.raises(ValueError):
            ens.get_species_dG(name="test",mu_dict=v)

    for v in variable_types["float_value_or_iter"]:
        print(v,type(v))
        mu_dict = {"X":v}
        ens = Ensemble()
        ens.add_species(name="test")
        ens.get_species_dG(name="test",mu_dict=mu_dict)

    for v in variable_types["not_float_value_or_iter"]:
        print(v,type(v))
    
        mu_dict = {"X":v}
        ens = Ensemble()
        ens.add_species(name="test")
        with pytest.raises(ValueError):
            ens.get_species_dG(name="test",mu_dict=mu_dict)


def test_Ensemble_get_obs(variable_types):

    # ------------------------------------------------------------------------
    # Not enough species
    ens = Ensemble()
    ens.add_species(name="test",
                    observable=False)
    with pytest.raises(ValueError):
        ens.get_obs()

    # ------------------------------------------------------------------------
    # No observable species
    ens = Ensemble()
    ens.add_species(name="test1",
                    observable=False)
    ens.add_species(name="test2",
                    observable=False)
    with pytest.raises(ValueError):
        ens.get_obs()

    # ------------------------------------------------------------------------
    # Only observable species
    ens = Ensemble()
    ens.add_species(name="test1",
                    observable=True)
    ens.add_species(name="test2",
                    observable=True)
    with pytest.raises(ValueError):
        ens.get_obs()

    # ------------------------------------------------------------------------
    # One observable, one not
    ens = Ensemble()
    ens.add_species(name="test1",
                    observable=True)
    ens.add_species(name="test2",
                    observable=False)
    
    df = ens.get_obs()
    assert df.loc[0,"dG_obs"] == 0
    assert df.loc[0,"fx_obs"] == 0.5
    assert df.loc[0,"test1"] == 0.5
    assert df.loc[0,"test2"] == 0.5

    # ------------------------------------------------------------------------
    # One observable, two not. (Use R = 1 and T = 1 to simplify math)
    ens = Ensemble(R=1)
    ens.add_species(name="test1",
                    observable=True)
    ens.add_species(name="test2",
                    observable=False)
    ens.add_species(name="test3",   
                    observable=False)
    
    df = ens.get_obs(T=1)
    assert df.loc[0,"dG_obs"] == -np.log(1/2)
    assert df.loc[0,"fx_obs"] == 1/3
    assert df.loc[0,"test1"] == 1/3
    assert df.loc[0,"test2"] == 1/3

    # ------------------------------------------------------------------------
    # One observable, two not. (Use R = 1 and T = 1 to simplify math). mu_dict
    # interesting.
    ens = Ensemble(R=1)
    ens.add_species(name="test1",
                    observable=True,
                    mu_stoich={"X":1})
    ens.add_species(name="test2",
                    observable=False)
    ens.add_species(name="test3",   
                    observable=False)
    df = ens.get_obs(T=1,
                     mu_dict={"X":[0,1]})
    
    assert df.loc[0,"dG_obs"] == -np.log(1/2)
    assert df.loc[0,"fx_obs"] == 1/3
    assert df.loc[0,"test1"] == 1/3
    assert df.loc[0,"test2"] == 1/3

    test1 = np.exp(-1)
    test2 = np.exp(0)
    test3 = np.exp(0)
    test_all = test1 + test2 + test3
    numerator = test1
    denominator = test2 + test3

    dG = -np.log(numerator/denominator)
    fx = numerator/(numerator + denominator)

    assert np.isclose(df.loc[1,"dG_obs"],dG)
    assert np.isclose(df.loc[1,"fx_obs"],fx)
    assert np.isclose(df.loc[1,"test1"],test1/test_all)
    assert np.isclose(df.loc[1,"test2"],test2/test_all)
    assert np.isclose(df.loc[1,"test3"],test3/test_all)

    # ------------------------------------------------------------------------
    # One observable, two not. (Use R = 1 and T = 1 to simplify math). mu_dict
    # interesting. mut_energy interesting
    ens = Ensemble(R=1)
    ens.add_species(name="test1",
                    observable=True,
                    mu_stoich={"X":1})
    ens.add_species(name="test2",
                    observable=False)
    ens.add_species(name="test3",   
                    observable=False)
    df = ens.get_obs(T=1,
                     mu_dict={"X":[0,1]},
                     mut_energy={"test2":-1})
    
    test1 = np.exp(-(0))
    test2 = np.exp(-(-1))
    test3 = np.exp(-(0))
    test_all = test1 + test2 + test3
    numerator = test1
    denominator = test2 + test3

    dG = -np.log(numerator/denominator)
    fx = numerator/(numerator + denominator)

    assert np.isclose(df.loc[0,"dG_obs"],dG)
    assert np.isclose(df.loc[0,"fx_obs"],fx)
    assert np.isclose(df.loc[0,"test1"],test1/test_all)
    assert np.isclose(df.loc[0,"test2"],test2/test_all)
    assert np.isclose(df.loc[0,"test3"],test3/test_all)

    test1 = np.exp(-(1))
    test2 = np.exp(-(-1))
    test3 = np.exp(-(0))
    test_all = test1 + test2 + test3
    numerator = test1
    denominator = test2 + test3

    dG = -np.log(numerator/denominator)
    fx = numerator/(numerator + denominator)

    assert np.isclose(df.loc[1,"dG_obs"],dG)
    assert np.isclose(df.loc[1,"fx_obs"],fx)
    assert np.isclose(df.loc[1,"test1"],test1/test_all)
    assert np.isclose(df.loc[1,"test2"],test2/test_all)
    assert np.isclose(df.loc[1,"test3"],test3/test_all)
    
    # ------------------------------------------------------------------------
    # One observable, two not. (Use R = 1 and T = 1 to simplify math). mu_dict
    # interesting, diff for different species. mut_energy interesting
    ens = Ensemble(R=1)
    ens.add_species(name="test1",
                    observable=True,
                    mu_stoich={"X":1})
    ens.add_species(name="test2",
                    observable=False,
                    mu_stoich={"Y":1})
    ens.add_species(name="test3",   
                    observable=False)
    df = ens.get_obs(T=1,
                     mu_dict={"X":[0,1],"Y":[0,1]},
                     mut_energy={"test2":-1})
    
    test1 = np.exp(-(0))
    test2 = np.exp(-(-1))
    test3 = np.exp(-(0))
    test_all = test1 + test2 + test3
    numerator = test1
    denominator = test2 + test3

    dG = -np.log(numerator/denominator)
    fx = numerator/(numerator + denominator)

    assert np.isclose(df.loc[0,"dG_obs"],dG)
    assert np.isclose(df.loc[0,"fx_obs"],fx)
    assert np.isclose(df.loc[0,"test1"],test1/test_all)
    assert np.isclose(df.loc[0,"test2"],test2/test_all)
    assert np.isclose(df.loc[0,"test3"],test3/test_all)

    test1 = np.exp(-(1))
    test2 = np.exp(-(0))
    test3 = np.exp(-(0))
    test_all = test1 + test2 + test3
    numerator = test1
    denominator = test2 + test3

    dG = -np.log(numerator/denominator)
    fx = numerator/(numerator + denominator)

    assert np.isclose(df.loc[1,"dG_obs"],dG)
    assert np.isclose(df.loc[1,"fx_obs"],fx)
    assert np.isclose(df.loc[1,"test1"],test1/test_all)
    assert np.isclose(df.loc[1,"test2"],test2/test_all)
    assert np.isclose(df.loc[1,"test3"],test3/test_all)


    # ------------------------------------------------------------------------
    # Overflow protection

    # this dG will overflow if put in raw but should be fine if we shift. 
    max_allowed = np.log(np.finfo("d").max)

    # Make sure this should overflow if used naively
    with pytest.warns(RuntimeWarning):
        x = np.exp(max_allowed + 1)
    assert np.isinf(x)

    ens = Ensemble(R=1)
    ens.add_species(name="test1",
                    observable=True,
                    dG0=max_allowed+2)
    ens.add_species(name="test2",
                    observable=False,
                    dG0=max_allowed+1)
    
    df = ens.get_obs(T=1)
    assert np.isclose(df.loc[0,"dG_obs"],-np.log(np.exp(-1)/np.exp(0)))
    assert np.isclose(df.loc[0,"fx_obs"],np.exp(-1)/(np.exp(0) + np.exp(-1)))
    assert np.isclose(df.loc[0,"test1"],np.exp(-1)/(np.exp(0) + np.exp(-1)))
    assert np.isclose( df.loc[0,"test2"],np.exp(0)/(np.exp(0) + np.exp(-1)))


    # mu_dict argument type checking
    print("--- mu_dict ---")
    for v in variable_types["dict"]:
        print(v,type(v))
        ens = Ensemble()
        ens.add_species(name="test1")
        ens.add_species(name="test2",observable=True)
        ens.get_obs(mu_dict=v)

    for v in variable_types["not_dict"]:
        print(v,type(v))

        # none okay
        if v is None:
            continue

        ens = Ensemble()
        ens.add_species(name="test1")
        ens.add_species(name="test2",observable=True)
        with pytest.raises(ValueError):
            ens.get_obs(mu_dict=v)

    for v in variable_types["float_value_or_iter"]:
        print(v,type(v))
        
        if hasattr(v,"__iter__") and len(v) == 0:
            continue

        if issubclass(type(v),pd.DataFrame):
            continue

        mu_dict = {"X":v}
        ens = Ensemble()
        ens.add_species(name="test1")
        ens.add_species(name="test2",observable=True)
        ens.get_obs(mu_dict=mu_dict)

    not_allowed = variable_types["not_float_value_or_iter"][:]
    not_allowed.append([])
    not_allowed.append(pd.DataFrame({"X":[1,2,3]}))

    for v in not_allowed:
        print(v,type(v))
    
        mu_dict = {"X":v}
        ens = Ensemble()
        ens.add_species(name="test1")
        ens.add_species(name="test2",observable=True)
        with pytest.raises(ValueError):
            ens.get_obs(mu_dict=mu_dict)

    print("--- mut_energy ---")
    for v in [{},{"test1":1},{"test2":1},{"test1":1,"test2":1}]: 
        print(v,type(v))

        ens = Ensemble()
        ens.add_species(name="test1")
        ens.add_species(name="test2",observable=True)
        ens.get_obs(mut_energy=v)

    not_allowed = variable_types["not_dict"]
    for v in not_allowed:
        print(v,type(v))
        if v is None:
            continue

        ens = Ensemble()
        ens.add_species(name="test1")
        ens.add_species(name="test2",observable=True)
        with pytest.raises(ValueError):
            ens.get_obs(mut_energy=v)

    for v in variable_types["floats_or_coercable"]:
        print(v,type(v))
        mut_energy = {"test1":v}

        ens = Ensemble()
        ens.add_species(name="test1")
        ens.add_species(name="test2",observable=True)
        ens.get_obs(mut_energy=mut_energy)
    

    for v in variable_types["not_floats_or_coercable"]:
        print(v,type(v))
        mut_energy = {"test1":v}

        ens = Ensemble()
        ens.add_species(name="test1")
        ens.add_species(name="test2",observable=True)
        with pytest.raises(ValueError):
            ens.get_obs(mut_energy=mut_energy)


    print("--- T ---")
    for v in [1,"1",1.0]:
        print(v,type(v))

        ens = Ensemble()
        ens.add_species(name="test1")
        ens.add_species(name="test2",observable=True)
        ens.get_obs(T=v)

    not_allowed = variable_types["not_floats_or_coercable"][:]
    not_allowed.append(0.0)
    not_allowed.append(-1.0)
    for v in not_allowed:
        print(v,type(v))

        ens = Ensemble()
        ens.add_species(name="test1")
        ens.add_species(name="test2",observable=True)
        with pytest.raises(ValueError):
            ens.get_obs(T=v)


def test_Ensemble_do_arg_checking():
    
    ens = Ensemble()

    assert ens.do_arg_checking == True
    ens.do_arg_checking = False
    assert ens.do_arg_checking == False

