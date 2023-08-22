import pytest

from eee.simulation.io.load_json import _validate_calc_kwargs
from eee.simulation.io.load_json import load_json
from eee.evolve.fitness.ff import ff_on
from eee.evolve.fitness.ff import ff_off

import numpy as np

import os
import shutil
import json
import copy


def test__validate_calc_kwargs():

    # No arg type/error checking. Internal function. 

    calc_type = "dummy" # not really used in function -- just for error message
    
    class TestClassSomeRequired:
        def __init__(self,
                     required_one,
                     required_two,
                     not_required_one=1,
                     not_required_two=2):
            """
            Docstring.
            """
            pass
            
    class TestClassAllRequired:
        def __init__(self,
                     required_one,
                     required_two):
            """
            Docstring.
            """
            pass
            
    class TestClassNoneRequired:
        def __init__(self,
                     not_required_one=1,
                     not_required_two=2):
            """
            Docstring.
            """
            pass
    
    # No args

    kwargs = {}
    with pytest.raises(ValueError):
        new_kwargs = _validate_calc_kwargs(calc_type=calc_type,
                                           calc_function=TestClassAllRequired.__init__,
                                           kwargs=kwargs)
    
    with pytest.raises(ValueError):
        new_kwargs = _validate_calc_kwargs(calc_type=calc_type,
                                           calc_function=TestClassSomeRequired.__init__,
                                           kwargs=kwargs)

    new_kwargs = _validate_calc_kwargs(calc_type=calc_type,
                                       calc_function=TestClassNoneRequired.__init__,
                                       kwargs=kwargs)
    assert new_kwargs is kwargs

    # One of the two required

    kwargs = {"required_one":1}
    with pytest.raises(ValueError):
        new_kwargs = _validate_calc_kwargs(calc_type=calc_type,
                                           calc_function=TestClassAllRequired.__init__,
                                           kwargs=kwargs)
    
    with pytest.raises(ValueError):
        new_kwargs = _validate_calc_kwargs(calc_type=calc_type,
                                           calc_function=TestClassSomeRequired.__init__,
                                           kwargs=kwargs)

    # Should now fail. Has unrecognized kwarg
    with pytest.raises(ValueError):
        new_kwargs = _validate_calc_kwargs(calc_type=calc_type,
                                           calc_function=TestClassNoneRequired.__init__,
                                           kwargs=kwargs)

    # Two of two required

    kwargs = {"required_one":1,
              "required_two":2}
    # Should now work. Have all required
    new_kwargs = _validate_calc_kwargs(calc_type=calc_type,
                                       calc_function=TestClassAllRequired.__init__,
                                       kwargs=kwargs)
    assert new_kwargs is kwargs
    
    # Should now work. Have all required
    new_kwargs = _validate_calc_kwargs(calc_type=calc_type,
                                       calc_function=TestClassSomeRequired.__init__,
                                       kwargs=kwargs)
    assert new_kwargs is kwargs

    # Should now fail. Has unrecognized kwarg
    with pytest.raises(ValueError):
        new_kwargs = _validate_calc_kwargs(calc_type=calc_type,
                                           calc_function=TestClassNoneRequired.__init__,
                                           kwargs=kwargs)

    # Two of two required, one optional

    kwargs = {"required_one":1,
              "required_two":2,
              "not_required_one":1}
    # Should now fail. Has unrecognized kwarg
    with pytest.raises(ValueError):
        new_kwargs = _validate_calc_kwargs(calc_type=calc_type,
                                           calc_function=TestClassAllRequired.__init__,
                                           kwargs=kwargs)
    
    # Should now work. Have all required
    new_kwargs = _validate_calc_kwargs(calc_type=calc_type,
                                       calc_function=TestClassSomeRequired.__init__,
                                       kwargs=kwargs)
    assert new_kwargs is kwargs

    # Should now fail. Has unrecognized kwarg
    with pytest.raises(ValueError):
        new_kwargs = _validate_calc_kwargs(calc_type=calc_type,
                                           calc_function=TestClassNoneRequired.__init__,
                                           kwargs=kwargs)
        
    # Two of two required, two optional

    kwargs = {"required_one":1,
              "required_two":2,
              "not_required_one":1,
              "not_required_two":2}
    # Should now fail. Has unrecognized kwarg
    with pytest.raises(ValueError):
        new_kwargs = _validate_calc_kwargs(calc_type=calc_type,
                                           calc_function=TestClassAllRequired.__init__,
                                           kwargs=kwargs)
    
    # Should now work. Have all required
    new_kwargs = _validate_calc_kwargs(calc_type=calc_type,
                                       calc_function=TestClassSomeRequired.__init__,
                                       kwargs=kwargs)
    assert new_kwargs is kwargs

    # Should now fail. Has unrecognized kwarg
    with pytest.raises(ValueError):
        new_kwargs = _validate_calc_kwargs(calc_type=calc_type,
                                           calc_function=TestClassNoneRequired.__init__,
                                           kwargs=kwargs)
        
    # Extra
    kwargs = {"required_one":1,
              "required_two":2,
              "not_required_one":1,
              "not_required_two":2,
              "extra":5}
    # Should now fail. Has unrecognized kwarg
    with pytest.raises(ValueError):
        new_kwargs = _validate_calc_kwargs(calc_type=calc_type,
                                           calc_function=TestClassAllRequired.__init__,
                                           kwargs=kwargs)
    
    # Should now fail. Has unrecognized kwarg
    with pytest.raises(ValueError):
        new_kwargs = _validate_calc_kwargs(calc_type=calc_type,
                                           calc_function=TestClassSomeRequired.__init__,
                                           kwargs=kwargs)

    # Should now fail. Has unrecognized kwarg
    with pytest.raises(ValueError):
        new_kwargs = _validate_calc_kwargs(calc_type=calc_type,
                                           calc_function=TestClassNoneRequired.__init__,
                                           kwargs=kwargs)


    # Only optional
    kwargs = {"not_required_one":1,
              "not_required_two":2}
    
    # Should now fail. Has unrecognized kwarg and missing arg
    with pytest.raises(ValueError):
        new_kwargs = _validate_calc_kwargs(calc_type=calc_type,
                                           calc_function=TestClassAllRequired.__init__,
                                           kwargs=kwargs)
    
    # Should now fail. Has missing kwargs
    with pytest.raises(ValueError):
        new_kwargs = _validate_calc_kwargs(calc_type=calc_type,
                                           calc_function=TestClassSomeRequired.__init__,
                                           kwargs=kwargs)

    # Should now work. Has matched optional args
    new_kwargs = _validate_calc_kwargs(calc_type=calc_type,
                                        calc_function=TestClassNoneRequired.__init__,
                                        kwargs=kwargs)
    assert new_kwargs is kwargs


def test_load_json(sim_json,test_ddg,tmpdir):

    current_dir = os.getcwd()
    os.chdir(tmpdir)

    shutil.copy(test_ddg["lac.csv"],"ddg.csv")

    sm, calc_params = load_json(sim_json["lac.json"],
                                use_stored_seed=False)

    species = ["hdna","h","l2e","unfolded"]
    assert np.array_equal(sm._ens.species,species)
    assert sm._ens._species["hdna"]["dG0"] == 0.0
    assert sm._ens._species["hdna"]["observable"] == True
    assert sm._ens._species["hdna"]["folded"] == True
    assert sm._ens._species["hdna"]["mu_stoich"] == {}

    assert sm._ens._species["h"]["dG0"] == 5.0
    assert sm._ens._species["h"]["observable"] == False
    assert sm._ens._species["h"]["folded"] == True
    assert sm._ens._species["h"]["mu_stoich"] == {}

    assert sm._ens._species["l2e"]["dG0"] == 5.0
    assert sm._ens._species["l2e"]["observable"] == False
    assert sm._ens._species["l2e"]["folded"] == True
    assert sm._ens._species["l2e"]["mu_stoich"] == {"iptg":4}

    assert sm._ens._species["unfolded"]["dG0"] == 10.0
    assert sm._ens._species["unfolded"]["observable"] == False
    assert sm._ens._species["unfolded"]["folded"] == False
    assert sm._ens._species["unfolded"]["mu_stoich"] == {}

    assert np.isclose(sm._ens._R,0.008314)

    assert np.array_equal(sm._fc.mu_dict["iptg"],np.array([1,4]))
    assert sm._fc._fitness_fcns[0] == ff_on
    assert sm._fc._fitness_fcns[1] == ff_off
    assert sm._fc._select_on == "dG_obs"
    assert sm._fc._select_on_folded == False
    assert sm._fc._fitness_kwargs == {}
    assert np.array_equal(sm._fc._T,[310.15,310.15])
    
    assert sm._gc._ddg_df.loc[0,"mut"] == "L1A"
    assert sm._seed != 487698321712

    assert len(calc_params) == 5
    assert calc_params["population_size"] == 100000
    assert np.isclose(calc_params["mutation_rate"],1e-5)
    assert calc_params["num_generations"] == 100000
    assert calc_params["write_prefix"] == "eee_sim_test"
    assert calc_params["write_frequency"] == 10000

    sm, calc_params = load_json(sim_json["lac.json"],
                                use_stored_seed=True)
    assert sm._seed == 487698321712

    with open(sim_json["lac.json"]) as f:
        template_json = json.load(f)

    test_json = copy.deepcopy(template_json)
    test_json["system"].pop("seed")
    with open('test.json','w') as f:
        json.dump(test_json,f)
    sm, calc_params = load_json("test.json",use_stored_seed=True)
    assert sm._seed != 487698321712

    test_json = copy.deepcopy(template_json)
    test_json.pop("calc_type")
    with open('test.json','w') as f:
        json.dump(test_json,f)
    with pytest.raises(ValueError):
        sm = load_json("test.json")
    
    test_json = copy.deepcopy(template_json)
    test_json.pop("calc_type")
    test_json["calc_type"] = "not_a_real_calc"
    with open('test.json','w') as f:
        json.dump(test_json,f)
    with pytest.raises(ValueError):
        sm = load_json("test.json")
    
    test_json = copy.deepcopy(template_json)
    test_json["system"].pop("ens")
    with open('test.json','w') as f:
        json.dump(test_json,f)
    with pytest.raises(ValueError):
        sm = load_json("test.json")

    test_json = copy.deepcopy(template_json)
    test_json["system"]["ens"].pop("R")
    with open('test.json','w') as f:
        json.dump(test_json,f)
    sm, calc_params = load_json("test.json")
    assert sm._ens._R == 0.001987

    test_json = copy.deepcopy(template_json)
    test_json["system"].pop("mu_dict")
    with open('test.json','w') as f:
        json.dump(test_json,f)
    with pytest.raises(ValueError):
        sm = load_json("test.json")

    test_json = copy.deepcopy(template_json)
    test_json["system"].pop("fitness_fcns")
    with open('test.json','w') as f:
        json.dump(test_json,f)
    with pytest.raises(ValueError):
        sm = load_json("test.json")
    
    test_json = copy.deepcopy(template_json)
    test_json["system"].pop("select_on")
    with open('test.json','w') as f:
        json.dump(test_json,f)
    sm, calc_params = load_json("test.json")
    assert sm._fc._select_on == "fx_obs"

    test_json = copy.deepcopy(template_json)
    test_json["system"].pop("select_on_folded")
    with open('test.json','w') as f:
        json.dump(test_json,f)
    sm, calc_params = load_json("test.json")
    assert sm._fc._select_on_folded == True

    test_json = copy.deepcopy(template_json)
    test_json["system"].pop("fitness_kwargs")
    with open('test.json','w') as f:
        json.dump(test_json,f)
    sm, calc_params = load_json("test.json")
    assert sm._fc._fitness_kwargs == {}

    test_json = copy.deepcopy(template_json)
    test_json["system"].pop("T")
    with open('test.json','w') as f:
        json.dump(test_json,f)
    sm, calc_params = load_json("test.json")
    assert np.array_equal(sm._fc._T,[298.15,298.15])

    test_json = copy.deepcopy(template_json)
    test_json["system"].pop("ddg_df")
    with open('test.json','w') as f:
        json.dump(test_json,f)
    with pytest.raises(ValueError):
        sm = load_json("test.json")

    test_json = copy.deepcopy(template_json)
    test_json["calc_params"].pop("population_size")
    with open('test.json','w') as f:
        json.dump(test_json,f)
    sm, calc_params = load_json("test.json")
    assert "population_size" not in calc_params
    assert len(calc_params) == 4

    test_json = copy.deepcopy(template_json)
    test_json["calc_params"].pop("mutation_rate")
    with open('test.json','w') as f:
        json.dump(test_json,f)
    sm, calc_params = load_json("test.json")
    assert "mutation_rate" not in calc_params
    assert len(calc_params) == 4

    test_json = copy.deepcopy(template_json)
    test_json["calc_params"].pop("num_generations")
    with open('test.json','w') as f:
        json.dump(test_json,f)
    sm, calc_params = load_json("test.json")
    assert "num_generations" not in calc_params
    assert len(calc_params) == 4

    test_json = copy.deepcopy(template_json)
    test_json["calc_params"].pop("write_prefix")
    with open('test.json','w') as f:
        json.dump(test_json,f)
    sm, calc_params = load_json("test.json")
    assert "write_prefix" not in calc_params
    assert len(calc_params) == 4

    test_json = copy.deepcopy(template_json)
    test_json["calc_params"].pop("write_frequency")
    with open('test.json','w') as f:
        json.dump(test_json,f)
    sm, calc_params = load_json("test.json")
    assert "write_frequency" not in calc_params
    assert len(calc_params) == 4

    os.chdir(current_dir)
