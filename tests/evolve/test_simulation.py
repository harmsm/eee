import pytest

from eee.evolve.simulation import SimulationContainer
from eee.evolve.simulation import load_json
from eee.evolve import FitnessContainer
from eee.evolve import GenotypeContainer
from eee.evolve.fitness import ff_on
from eee.evolve.fitness import ff_off

import numpy as np

import os
import shutil
import json
import copy


def test_SimulationContainer_load_json(sim_json,test_ddg,tmpdir):

    current_dir = os.getcwd()
    os.chdir(tmpdir)

    shutil.copy(test_ddg["lac.csv"],"ddg.csv")

    sm = load_json(sim_json["lac.json"],
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
    
    assert sm._population_size == 100000
    assert np.isclose(sm._mutation_rate,1e-5)
    assert sm._num_generations == 100000
    assert sm._write_prefix == "eee_sim_test"
    assert sm._write_frequency == 10000
    assert sm._seed != 487698321712
    
    sm = load_json(sim_json["lac.json"],
                   use_stored_seed=True)
    assert sm._seed == 487698321712

    with open(sim_json["lac.json"]) as f:
        template_json = json.load(f)
    
    test_json = copy.deepcopy(template_json)
    test_json.pop("ens")
    with open('test.json','w') as f:
        json.dump(test_json,f)
    with pytest.raises(ValueError):
        sm = load_json("test.json")

    test_json = copy.deepcopy(template_json)
    test_json.pop("R")
    with open('test.json','w') as f:
        json.dump(test_json,f)
    sm = load_json("test.json")
    assert sm._ens._R == 0.001987

    test_json = copy.deepcopy(template_json)
    test_json.pop("mu_dict")
    with open('test.json','w') as f:
        json.dump(test_json,f)
    with pytest.raises(ValueError):
        sm = load_json("test.json")

    test_json = copy.deepcopy(template_json)
    test_json.pop("fitness_fcns")
    with open('test.json','w') as f:
        json.dump(test_json,f)
    with pytest.raises(ValueError):
        sm = load_json("test.json")
    
    test_json = copy.deepcopy(template_json)
    test_json.pop("select_on")
    with open('test.json','w') as f:
        json.dump(test_json,f)
    sm = load_json("test.json")
    assert sm._fc._select_on == "fx_obs"

    test_json = copy.deepcopy(template_json)
    test_json.pop("select_on_folded")
    with open('test.json','w') as f:
        json.dump(test_json,f)
    sm = load_json("test.json")
    assert sm._fc._select_on_folded == True

    test_json = copy.deepcopy(template_json)
    test_json.pop("select_on")
    with open('test.json','w') as f:
        json.dump(test_json,f)
    sm = load_json("test.json")
    assert sm._fc._fitness_kwargs == {}

    test_json = copy.deepcopy(template_json)
    test_json.pop("T")
    with open('test.json','w') as f:
        json.dump(test_json,f)
    sm = load_json("test.json")
    assert np.array_equal(sm._fc._T,[298.15,298.15])


    test_json = copy.deepcopy(template_json)
    test_json.pop("ddg_df")
    with open('test.json','w') as f:
        json.dump(test_json,f)
    with pytest.raises(ValueError):
        sm = load_json("test.json")

    test_json = copy.deepcopy(template_json)
    test_json.pop("population_size")
    with open('test.json','w') as f:
        json.dump(test_json,f)
    sm = load_json("test.json")
    sm._population_size == 1000

    test_json = copy.deepcopy(template_json)
    test_json.pop("mutation_rate")
    with open('test.json','w') as f:
        json.dump(test_json,f)
    sm = load_json("test.json")
    sm._mutation_rate == 0.01

    test_json = copy.deepcopy(template_json)
    test_json.pop("num_generations")
    with open('test.json','w') as f:
        json.dump(test_json,f)
    sm = load_json("test.json")
    sm._mutation_rate == 100

    test_json = copy.deepcopy(template_json)
    test_json.pop("write_prefix")
    with open('test.json','w') as f:
        json.dump(test_json,f)
    sm = load_json("test.json")
    sm._write_prefix == "eee_sim"

    test_json = copy.deepcopy(template_json)
    test_json.pop("write_frequency")
    with open('test.json','w') as f:
        json.dump(test_json,f)
    sm = load_json("test.json")
    sm._write_frequency == 1000

    test_json = copy.deepcopy(template_json)
    test_json.pop("seed")
    with open('test.json','w') as f:
        json.dump(test_json,f)
    sm = load_json("test.json",use_stored_seed=True)
    assert sm._seed != 487698321712

    os.chdir(current_dir)


def test_SimulationContainer(ens_test_data,variable_types):

    ens = ens_test_data["ens"]
    ddg_df = ens_test_data["ddg_df"]
    mu_dict = ens_test_data["mu_dict"]
    fitness_fcns = ens_test_data["fitness_fcns"]

    # ens
    for v in variable_types["everything"]:
        print(v,type(v),flush=True)
        with pytest.raises(ValueError):
            sm = SimulationContainer(ens=v,
                                     ddg_df=ddg_df,
                                     mu_dict=mu_dict,
                                     fitness_fcns=fitness_fcns,
                                     select_on="fx_obs",
                                     fitness_kwargs={},
                                     T=1,
                                     population_size=100,
                                     mutation_rate=0.01,
                                     num_generations=100,
                                     write_prefix="eee_sim",
                                     write_frequency=1000,
                                     seed=None)
        
    # ddg_df
    for v in variable_types["everything"]:

        # skip strings -- leads to FileNotFound error because attempt to read
        # str as file
        if issubclass(type(v),str):
            continue

        print(v,type(v),flush=True)
        with pytest.raises(ValueError):
            sm = SimulationContainer(ens=ens,
                                     ddg_df=v,
                                     mu_dict=mu_dict,
                                     fitness_fcns=fitness_fcns,
                                     select_on="fx_obs",
                                     fitness_kwargs={},
                                     T=1,
                                     population_size=100,
                                     mutation_rate=0.01,
                                     num_generations=100,
                                     write_prefix="eee_sim",
                                     write_frequency=1000,
                                     seed=None)

    # mu_dict
    for v in variable_types["everything"]:
        print(v,type(v),flush=True)
        with pytest.raises(ValueError):
            sm = SimulationContainer(ens=ens,
                                     ddg_df=ddg_df,
                                     mu_dict=v,
                                     fitness_fcns=fitness_fcns,
                                     select_on="fx_obs",
                                     fitness_kwargs={},
                                     T=1,
                                     population_size=100,
                                     mutation_rate=0.01,
                                     num_generations=100,
                                     write_prefix="eee_sim",
                                     write_frequency=1000,
                                     seed=None)

    # fitness_fcns
    for v in variable_types["everything"]:
        print(v,type(v),flush=True)
        with pytest.raises(ValueError):
            sm = SimulationContainer(ens=ens,
                                     ddg_df=ddg_df,
                                     mu_dict=mu_dict,
                                     fitness_fcns=v,
                                     select_on="fx_obs",
                                     fitness_kwargs={},
                                     T=1,
                                     population_size=100,
                                     mutation_rate=0.01,
                                     num_generations=100,
                                     write_prefix="eee_sim",
                                     write_frequency=1000,
                                     seed=None)

    # select_on
    for v in variable_types["everything"]:
        print(v,type(v),flush=True)
        with pytest.raises(ValueError):
            sm = SimulationContainer(ens=ens,
                                     ddg_df=ddg_df,
                                     mu_dict=mu_dict,
                                     fitness_fcns=fitness_fcns,
                                     select_on=v,
                                     fitness_kwargs={},
                                     T=1,
                                     population_size=100,
                                     mutation_rate=0.01,
                                     num_generations=100,
                                     write_prefix="eee_sim",
                                     write_frequency=1000,
                                     seed=None)
            
    sm = SimulationContainer(ens=ens,
                             ddg_df=ddg_df,
                             mu_dict=mu_dict,
                             fitness_fcns=fitness_fcns,
                             select_on="dG_obs",
                             fitness_kwargs={},
                             T=1,
                             population_size=100,
                             mutation_rate=0.01,
                             num_generations=100,
                             write_prefix="eee_sim",
                             write_frequency=1000,
                             seed=None)
    
    # select_on_folded
    for v in variable_types["not_bools"]:
        print(v,type(v),flush=True)
        with pytest.raises(ValueError):
            sm = SimulationContainer(ens=ens,
                                     ddg_df=ddg_df,
                                     mu_dict=mu_dict,
                                     fitness_fcns=fitness_fcns,
                                     select_on="fx_obs",
                                     select_on_folded=v,
                                     fitness_kwargs={},
                                     T=1,
                                     population_size=100,
                                     mutation_rate=0.01,
                                     num_generations=100,
                                     write_prefix="eee_sim",
                                     write_frequency=1000,
                                     seed=None)
            
    sm = SimulationContainer(ens=ens,
                             ddg_df=ddg_df,
                             mu_dict=mu_dict,
                             fitness_fcns=fitness_fcns,
                             select_on="dG_obs",
                             fitness_kwargs={},
                             T=1,
                             population_size=100,
                             mutation_rate=0.01,
                             num_generations=100,
                             write_prefix="eee_sim",
                             write_frequency=1000,
                             seed=None)

    # fitness_kwargs
    for v in variable_types["everything"]:
        if issubclass(type(v),dict):
            continue
        if v is None:
            continue

        print(v,type(v),flush=True)
        with pytest.raises(ValueError):
            sm = SimulationContainer(ens=ens,
                                     ddg_df=ddg_df,
                                     mu_dict=mu_dict,
                                     fitness_fcns=fitness_fcns,
                                     select_on="fx_obs",
                                     fitness_kwargs=v,
                                     T=1,
                                     population_size=100,
                                     mutation_rate=0.01,
                                     num_generations=100,
                                     write_prefix="eee_sim",
                                     write_frequency=1000,
                                     seed=None)

    # T
    for v in variable_types["everything"]:
        
        # Skip coercable to float values
        try:
            float_v = float(v)
            if float_v > 0:
                continue
        except:
            pass

        print(v,type(v),flush=True)
        with pytest.raises(ValueError):
            sm = SimulationContainer(ens=ens,
                                     ddg_df=ddg_df,
                                     mu_dict=mu_dict,
                                     fitness_fcns=fitness_fcns,
                                     select_on="fx_obs",
                                     fitness_kwargs={},
                                     T=v,
                                     population_size=100,
                                     mutation_rate=0.01,
                                     num_generations=100,
                                     write_prefix="eee_sim",
                                     write_frequency=1000,
                                     seed=None)

    # seed
    for v in variable_types["everything"]:
        
        # Skip coercable to float values
        try:
            int_v = int(v)
            if int_v >= 0:
                continue
        except:
            pass

        if v is None:
            continue

        print(v,type(v),flush=True)
        with pytest.raises(ValueError):
            sm = SimulationContainer(ens=ens,
                                     ddg_df=ddg_df,
                                     mu_dict=mu_dict,
                                     fitness_fcns=fitness_fcns,
                                     select_on="fx_obs",
                                     fitness_kwargs={},
                                     T=1,
                                     population_size=100,
                                     mutation_rate=0.01,
                                     num_generations=100,
                                     write_prefix="eee_sim",
                                     write_frequency=1000,
                                     seed=v)

    # Now test that things are being set reasonably well

    # ------------------------------------------------------------------
    # Check seed
    
    sm = SimulationContainer(ens=ens,
                             ddg_df=ddg_df,
                             mu_dict=mu_dict,
                             fitness_fcns=fitness_fcns,
                             select_on="fx_obs",
                             fitness_kwargs={},
                             T=1,
                             population_size=100,
                             mutation_rate=0.01,
                             num_generations=100,
                             write_prefix="eee_sim",
                             write_frequency=1000,
                             seed=None)
    
    assert issubclass(type(sm._seed),int)
    assert sm._seed >= 0
    assert issubclass(type(sm._pcg64),np.random._pcg64.PCG64)

    sm = SimulationContainer(ens=ens,
                             ddg_df=ddg_df,
                             mu_dict=mu_dict,
                             fitness_fcns=fitness_fcns,
                             select_on="fx_obs",
                             fitness_kwargs={},
                             T=1,
                             population_size=100,
                             mutation_rate=0.01,
                             num_generations=100,
                             write_prefix="eee_sim",
                             write_frequency=1000,
                             seed=5)

    assert issubclass(type(sm._seed),int)
    assert sm._seed == 5
    assert issubclass(type(sm._pcg64),np.random._pcg64.PCG64)
    assert np.isclose(0.8050029237453802,sm._rng.random())

    assert issubclass(type(sm._fc),FitnessContainer)
    assert issubclass(type(sm._gc),GenotypeContainer)



def test_SimulationContainer_run(ens_test_data,tmpdir):

    # BETTER TESTING CHECKING XXXX

    current_dir = os.getcwd()
    os.chdir(tmpdir)

    ens = ens_test_data["ens"]
    ddg_df = ens_test_data["ddg_df"]
    mu_dict = ens_test_data["mu_dict"]
    fitness_fcns = ens_test_data["fitness_fcns"]
    
    sm = SimulationContainer(ens=ens,
                             ddg_df=ddg_df,
                             mu_dict=mu_dict,
                             fitness_fcns=fitness_fcns,
                             select_on="fx_obs",
                             fitness_kwargs={},
                             T=1,
                             population_size=100,
                             mutation_rate=0.01,
                             num_generations=100,
                             write_prefix="eee_sim",
                             write_frequency=1000,
                             seed=None)
    
    sm.run("test")
    assert os.path.exists(os.path.join("test","ddg.csv"))
    assert os.path.exists(os.path.join("test","simulation.json"))
    assert os.path.exists(os.path.join("test","eee_sim_genotypes.csv"))
    assert os.path.exists(os.path.join("test","eee_sim_generations_0.pickle"))
 
    os.chdir(current_dir)

def test_SimulationContainer_to_dict(ens_test_data):
    
    ens = ens_test_data["ens"]
    ddg_df = ens_test_data["ddg_df"]
    mu_dict = ens_test_data["mu_dict"]
    fitness_fcns = ens_test_data["fitness_fcns"]

    sm = SimulationContainer(ens=ens,
                             ddg_df=ddg_df,
                             mu_dict=mu_dict,
                             fitness_fcns=fitness_fcns,
                             select_on="fx_obs",
                             fitness_kwargs={},
                             T=1,
                             population_size=100,
                             mutation_rate=0.01,
                             num_generations=100,
                             write_prefix="eee_sim",
                             write_frequency=1000,
                             seed=5)
    
    out = sm.to_dict()
    assert out["population_size"] == 100
    assert out["mutation_rate"] == 0.01
    assert out["num_generations"] == 100
    assert out["write_prefix"] == "eee_sim"
    assert out["write_frequency"] == 1000
    assert out["seed"] == 5


def test_SimulationContainer_write_json():
    pass
