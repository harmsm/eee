import pytest

from eee.simulation.core.simulation import Simulation
from eee.simulation.core import Fitness
from eee.simulation.core import Genotype

import numpy as np
import pandas as pd

import os
import json

class Simulation_no_run(Simulation):
    """
    Bad dummy class. Does not have run defined, so it should throw 
    NotImplementedError.
    """

    calc_type = "fake_no_run"

class Simulation_no_calc_type(Simulation):
    """
    Bad dummy class. Does not have calc_type defined, so it should throw 
    NotImplementedError.
    """

    def run(self,*args,**kwargs):
        pass

class SimulationTester(Simulation):
    """
    Simulation cannot be run on its own. This dummy subclass can 
    be used to test its core functionality.
    """

    calc_type = "fake"

    def run(self,
            output_directory,
            population_size=100,
            mutation_rate=0.01,
            num_generations=100,
            write_prefix="eee_sim",
            write_frequency=1000):
        
        calc_params = {"population_size":population_size,
                       "mutation_rate":mutation_rate,
                       "num_generations":num_generations,
                       "write_prefix":write_prefix,
                       "write_frequency":write_frequency}

        self._prepare_calc(output_directory=output_directory,
                           calc_params=calc_params)
        
        f = open("output_file.txt","w")
        f.write("fake\n")
        f.close()

        self._complete_calc()

def test_Simulation(ens_test_data,variable_types):

    ens = ens_test_data["ens"]
    ddg_df = ens_test_data["ddg_df"]
    mu_dict = ens_test_data["mu_dict"]
    fitness_fcns = ens_test_data["fitness_fcns"]

    # Make sure implementation checks are in place. 
    with pytest.raises(NotImplementedError):
        sm = Simulation(ens=ens,
                                 ddg_df=ddg_df,
                                 mu_dict=mu_dict,
                                 fitness_fcns=fitness_fcns,
                                 select_on="fx_obs",
                                 fitness_kwargs={},
                                 T=1,
                                 seed=None)


    with pytest.raises(NotImplementedError):
        sm = Simulation_no_run(ens=ens,
                               ddg_df=ddg_df,
                               mu_dict=mu_dict,
                               fitness_fcns=fitness_fcns,
                               select_on="fx_obs",
                               fitness_kwargs={},
                               T=1,
                               seed=None)

    with pytest.raises(NotImplementedError):
        sm = Simulation_no_calc_type(ens=ens,
                                     ddg_df=ddg_df,
                                     mu_dict=mu_dict,
                                     fitness_fcns=fitness_fcns,
                                     select_on="fx_obs",
                                     fitness_kwargs={},
                                     T=1,
                                     seed=None)


    # Now test variable checking

    # ens
    for v in variable_types["everything"]:
        print(v,type(v),flush=True)
        with pytest.raises(ValueError):
            sm = SimulationTester(ens=v,
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
            sm = SimulationTester(ens=ens,
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
            sm = SimulationTester(ens=ens,
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
            sm = SimulationTester(ens=ens,
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
            sm = SimulationTester(ens=ens,
                                           ddg_df=ddg_df,
                                           mu_dict=mu_dict,
                                           fitness_fcns=fitness_fcns,
                                           select_on=v,
                                           fitness_kwargs={},
                                           T=1,
                                           seed=None)
                   
    sm = SimulationTester(ens=ens,
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
            sm = SimulationTester(ens=ens,
                                           ddg_df=ddg_df,
                                           mu_dict=mu_dict,
                                           fitness_fcns=fitness_fcns,
                                           select_on="fx_obs",
                                           select_on_folded=v,
                                           fitness_kwargs={},
                                           T=1,
                                           seed=None)
            
    sm = SimulationTester(ens=ens,
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
            sm = SimulationTester(ens=ens,
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
            sm = SimulationTester(ens=ens,
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
            sm = SimulationTester(ens=ens,
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
    
    sm = SimulationTester(ens=ens,
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

    sm = SimulationTester(ens=ens,
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

    assert issubclass(type(sm._fc),Fitness)
    assert issubclass(type(sm._gc),Genotype)

def test_Simulation__prepare_calc(ens_test_data,tmpdir):

    current_dir = os.getcwd()
    os.chdir(tmpdir)

    assert not os.path.exists("test_dir")

    ens = ens_test_data["ens"]
    ddg_df = ens_test_data["ddg_df"]
    mu_dict = ens_test_data["mu_dict"]
    fitness_fcns = ens_test_data["fitness_fcns"]


    sm = SimulationTester(ens=ens,
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

def test_Simulation__write_calc_params(ens_test_data,
                                       tmpdir):

    current_dir = os.getcwd()
    os.chdir(tmpdir)

    ens = ens_test_data["ens"]
    ddg_df = ens_test_data["ddg_df"]
    mu_dict = ens_test_data["mu_dict"]
    fitness_fcns = ens_test_data["fitness_fcns"]

    sm = SimulationTester(ens=ens,
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

    assert ens_dict["ens"]["gas_constant"] == as_written["system"]["ens"]["gas_constant"]

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

def test_Simulation__complete_calc():
    # Tested within test__prepare_calc because these are paired functions. 
    assert True
    
def test_Simulation_system_params(ens_test_data):
    
    ens = ens_test_data["ens"]
    ddg_df = ens_test_data["ddg_df"]
    mu_dict = ens_test_data["mu_dict"]
    fitness_fcns = ens_test_data["fitness_fcns"]

    sm = SimulationTester(ens=ens,
                          ddg_df=ddg_df,
                          mu_dict=mu_dict,
                          fitness_fcns=fitness_fcns,
                          select_on="fx_obs",
                          select_on_folded=True,
                          fitness_kwargs={},
                          T=1,
                          seed=5)
    
    system_params = sm.system_params

    ens = sm._ens.to_dict()
    for k in system_params["ens"]:
        assert ens["ens"][k] == system_params["ens"][k]

    fc = sm._fc.to_dict()
    for k in fc:
        if hasattr(fc[k],"__iter__"):
            assert np.array_equal(fc[k],system_params[k])
        else:
            assert fc[k] == system_params[k]

    gc = sm._gc.to_dict()
    for k in gc:
        if hasattr(gc[k],"__iter__"):
            assert np.array_equal(fc[k],system_params[k])
        else:
            assert gc[k] == system_params[k]

    assert system_params["seed"] == sm._seed


def test_Simulation_get_calc_description(ens_test_data):
    
    # print not a great test... mostly just make sure it runs without error and
    # produces a string. 

    ens = ens_test_data["ens"]
    ddg_df = ens_test_data["ddg_df"]
    mu_dict = ens_test_data["mu_dict"]
    fitness_fcns = ens_test_data["fitness_fcns"]

    sm = SimulationTester(ens=ens,
                          ddg_df=ddg_df,
                          mu_dict=mu_dict,
                          fitness_fcns=fitness_fcns,
                          select_on="fx_obs",
                          select_on_folded=True,
                          fitness_kwargs={},
                          T=1,
                          seed=5)
    
    # No kwargs passed in 
    value1 = sm.get_calc_description()
    assert issubclass(type(value1),str)

    # Kwargs passed in
    value2 = sm.get_calc_description(calc_kwargs={"mutation_rate":0.1})
    assert issubclass(type(value2),str)

    # kwargs should make longer
    assert len(value1) < len(value2)


def test_Simulation_fitness_from_energy(ens_test_data):
    
    ens = ens_test_data["ens"]
    ddg_df = ens_test_data["ddg_df"]
    mu_dict = ens_test_data["mu_dict"]
    fitness_fcns = ens_test_data["fitness_fcns"]

    sm = SimulationTester(ens=ens,
                          ddg_df=ddg_df,
                          mu_dict=mu_dict,
                          fitness_fcns=fitness_fcns,
                          select_on="fx_obs",
                          select_on_folded=True,
                          fitness_kwargs={},
                          T=1)
    
    f = sm.fitness_from_energy({"s1":0,"s2":0})
    assert f == np.prod(sm._gc._fitness_function(np.zeros(2)))

    f = sm.fitness_from_energy(np.zeros(2))
    assert f == np.prod(sm._gc._fitness_function(np.zeros(2)))

    
    mut_energy = np.array([1,0])
    f = sm.fitness_from_energy({"s1":1,"s2":0})
    assert f == np.prod(sm._gc._fitness_function(mut_energy))

    f = sm.fitness_from_energy(mut_energy)
    assert f == np.prod(sm._gc._fitness_function(mut_energy))

def test_Simulation_ens(ens_test_data):
    
    ens = ens_test_data["ens"]
    ddg_df = ens_test_data["ddg_df"]
    mu_dict = ens_test_data["mu_dict"]
    fitness_fcns = ens_test_data["fitness_fcns"]

    sm = SimulationTester(ens=ens,
                          ddg_df=ddg_df,
                          mu_dict=mu_dict,
                          fitness_fcns=fitness_fcns,
                          select_on="fx_obs",
                          select_on_folded=True,
                          fitness_kwargs={},
                          T=1)
    
    assert ens is sm.ens
    assert ens is sm._ens

def test_Simulation_fc(ens_test_data):
    
    ens = ens_test_data["ens"]
    ddg_df = ens_test_data["ddg_df"]
    mu_dict = ens_test_data["mu_dict"]
    fitness_fcns = ens_test_data["fitness_fcns"]

    sm = SimulationTester(ens=ens,
                          ddg_df=ddg_df,
                          mu_dict=mu_dict,
                          fitness_fcns=fitness_fcns,
                          select_on="fx_obs",
                          select_on_folded=True,
                          fitness_kwargs={},
                          T=1)
    
    assert sm.fc is sm._fc

def test_Simulation_gc(ens_test_data):
    
    ens = ens_test_data["ens"]
    ddg_df = ens_test_data["ddg_df"]
    mu_dict = ens_test_data["mu_dict"]
    fitness_fcns = ens_test_data["fitness_fcns"]

    sm = SimulationTester(ens=ens,
                          ddg_df=ddg_df,
                          mu_dict=mu_dict,
                          fitness_fcns=fitness_fcns,
                          select_on="fx_obs",
                          select_on_folded=True,
                          fitness_kwargs={},
                          T=1)
    
    assert sm.gc is sm._gc