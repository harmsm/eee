import pytest

from eee.evolve.simulation.container import SimulationContainer
from eee.evolve import FitnessContainer
from eee.evolve import GenotypeContainer

import numpy as np
import pandas as pd

import os
import json

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
                                     seed=None)
            
    sm = SimulationContainer(ens=ens,
                             ddg_df=ddg_df,
                             mu_dict=mu_dict,
                             fitness_fcns=fitness_fcns,
                             select_on="dG_obs",
                             fitness_kwargs={},
                             T=1,
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
                                     seed=None)
            
    sm = SimulationContainer(ens=ens,
                             ddg_df=ddg_df,
                             mu_dict=mu_dict,
                             fitness_fcns=fitness_fcns,
                             select_on="dG_obs",
                             fitness_kwargs={},
                             T=1,
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
                             seed=5)

    assert issubclass(type(sm._seed),int)
    assert sm._seed == 5
    assert issubclass(type(sm._pcg64),np.random._pcg64.PCG64)
    assert np.isclose(0.8050029237453802,sm._rng.random())

    assert issubclass(type(sm._fc),FitnessContainer)
    assert issubclass(type(sm._gc),GenotypeContainer)

def test_SimulationContainer__prepare_calc(ens_test_data,tmpdir):

    current_dir = os.getcwd()
    os.chdir(tmpdir)

    assert not os.path.exists("test_dir")

    ens = ens_test_data["ens"]
    ddg_df = ens_test_data["ddg_df"]
    mu_dict = ens_test_data["mu_dict"]
    fitness_fcns = ens_test_data["fitness_fcns"]


    sm = SimulationContainer(ens=ens,
                            ddg_df=ddg_df,
                            mu_dict=mu_dict,
                            fitness_fcns=fitness_fcns,
                            select_on="fx_obs",
                            select_on_folded=True,
                            fitness_kwargs={},
                            T=1,
                            seed=5)
    
    sm._prepare_calc(output_directory="test_dir",
                     calc_params={"test":1})
    assert os.path.split(os.getcwd())[-1] == "test_dir"
    with open("simulation.json") as f:
        sim_json = json.load(f)
    assert sim_json["calc_params"]["test"] == 1

    sm._complete_calc()
    assert os.getcwd() == tmpdir
    assert os.path.exists("test_dir")

    # Should not work now because directory exists
    with pytest.raises(FileExistsError):
        sm._prepare_calc(output_directory="test_dir",
                         calc_params={"test":1})


    os.chdir(current_dir)

def test_SimulationContainer__write_calc_params(ens_test_data,
                                                tmpdir):

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
                             select_on_folded=True,
                             fitness_kwargs={},
                             T=1,
                             seed=5)
    
    calc_params = {"population_size":100,
                  "mutation_rate":0.01,
                  "num_generations":100,
                  "write_prefix":"eee_sim",
                  "write_frequency":1000}
    
    sm._write_calc_params(calc_params=calc_params)

    assert os.path.exists("simulation.json")
    assert os.path.exists("ddg.csv")

    with open("simulation.json") as f:
        as_written = json.load(f)

    # Make sure run params written correctly. 
    for k in calc_params:
        assert as_written["calc_params"][k] == calc_params[k]
    
    # Make sure ensemble written correctly
    ens_dict = ens.to_dict()
    for k in ens_dict["ens"]:
        assert ens_dict["ens"][k] == as_written["system"]["ens"][k]

    assert ens_dict["ens"]["R"] == as_written["system"]["ens"]["R"]

    assert as_written["system"]["mu_dict"] == mu_dict
    assert as_written["system"]["select_on"] == "fx_obs"
    assert as_written["system"]["select_on_folded"] == True
    assert as_written["system"]["fitness_kwargs"] == {}
    assert np.array_equal(as_written["system"]["T"],[1,1])
    assert np.array_equal(as_written["system"]["fitness_fcns"],
                          ["on","off"])
    assert as_written["system"]["ddg_df"] == "ddg.csv"
    
    # Make sure we can read in dataframe
    df = pd.read_csv("ddg.csv")

    os.chdir(current_dir)

def test_SimulationContainer__complete_calc():
    # Tested within test__prepare_calc because these are paired functions. 
    assert True
    

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
                             seed=None)
    
    sm.run(output_directory="test",
           population_size=100,
           mutation_rate=0.01,
           num_generations=100,
           write_prefix="eee_sim",
           write_frequency=1000)
    assert os.path.exists(os.path.join("test","ddg.csv"))
    assert os.path.exists(os.path.join("test","simulation.json"))
    assert os.path.exists(os.path.join("test","eee_sim_genotypes.csv"))
    assert os.path.exists(os.path.join("test","eee_sim_generations_0.pickle"))
 
    os.chdir(current_dir)
