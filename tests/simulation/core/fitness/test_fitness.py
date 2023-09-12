
from eee import Ensemble

from eee.simulation.core.fitness.fitness import Fitness
from eee.simulation.core.fitness.ff import ff_on
from eee.simulation.core.fitness.ff import ff_off

import numpy as np
import pandas as pd

import pytest

def test_Fitness(ens_test_data,variable_types):

    # Basic ensemble    
    ens = Ensemble(gas_constant=1)
    ens.add_species(name="test1",
                    observable=True,
                    folded=False,
                    ligand_stoich={"X":1})
    ens.add_species(name="test2",
                    observable=False,
                    folded=True,
                    ligand_stoich={"Y":1})
    
    ligand_dict = {"X":[0,10000],"Y":[10000,0]}
    fitness_fcns = [ff_off,ff_on]
    select_on = "fx_obs"
    select_on_folded = True
    temperature = 1

    # make sure attributes are set correctly
    fc = Fitness(ens=ens,
                          ligand_dict=ligand_dict,
                          fitness_fcns=fitness_fcns,
                          select_on=select_on,
                          select_on_folded=select_on_folded,
                          fitness_kwargs={},
                          temperature=temperature)
  
    # fc._private_ens should be a copy of ens
    assert fc.ens is ens
    assert fc._private_ens is not ens

    assert np.array_equal(fc.temperature,[1.0,1.0])
    
    for k in fc._ligand_dict:
        assert np.array_equal(ligand_dict[k],fc._ligand_dict[k])
    assert fc._fitness_fcns[0] == fitness_fcns[0]
    assert fc._fitness_fcns[1] == fitness_fcns[1]
    assert fc._select_on == select_on
    assert fc._select_on_folded == select_on_folded
    assert len(fc._fitness_kwargs) == 0
    assert issubclass(type(fc._fitness_kwargs),dict)
    assert np.array_equal(fc.temperature,np.ones(2,dtype=float))

    ## Check variable type checking

    ens = ens_test_data["ens"]
    ligand_dict = ens_test_data["ligand_dict"]
    fitness_fcns = ens_test_data["fitness_fcns"]

    # ens
    for v in variable_types["everything"]:
        print(v,type(v),flush=True)    
        with pytest.raises(ValueError):
            Fitness(ens=v,
                             ligand_dict=ligand_dict,
                             fitness_fcns=fitness_fcns,
                             select_on="fx_obs",
                             select_on_folded=True,
                             fitness_kwargs={},
                             temperature=1)
        
    # ligand_dict
    for v in variable_types["everything"]:
        print(v,type(v),flush=True)
        with pytest.raises(ValueError):
            Fitness(ens=ens,
                             ligand_dict=v,
                             fitness_fcns=fitness_fcns,
                             select_on="fx_obs",
                             select_on_folded=True,
                             fitness_kwargs={},
                             temperature=1)

    # fitness_fcns
    for v in variable_types["everything"]:
        print(v,type(v),flush=True)
        with pytest.raises(ValueError):
            Fitness(ens=ens,
                             ligand_dict=ligand_dict,
                             fitness_fcns=v,
                             select_on="fx_obs",
                             select_on_folded=True,
                             fitness_kwargs={},
                             temperature=1)
            
    # select_on
    for v in variable_types["everything"]:
        print(v,type(v),flush=True)
        with pytest.raises(ValueError):
            Fitness(ens=ens,
                             ligand_dict=ligand_dict,
                             fitness_fcns=fitness_fcns,
                             select_on=v,
                             select_on_folded=True,
                             fitness_kwargs={},
                             temperature=1)
            
    Fitness(ens=ens,
                     ligand_dict=ligand_dict,
                     fitness_fcns=fitness_fcns,
                     select_on="dG_obs",
                     select_on_folded=True,
                     fitness_kwargs={},
                     temperature=1)
    
    # select_on_folded
    for v in variable_types["not_bools"]:
        print(v,type(v),flush=True)
        with pytest.raises(ValueError):
            Fitness(ens=ens,
                             ligand_dict=ligand_dict,
                             fitness_fcns=fitness_fcns,
                             select_on="fx_obs",
                             select_on_folded=v,
                             fitness_kwargs={},
                             temperature=1)

    # fitness_kwargs
    for v in variable_types["everything"]:
        if issubclass(type(v),dict):
            continue
        if v is None:
            continue

        print(v,type(v),flush=True)
        with pytest.raises(ValueError):
            Fitness(ens=ens,
                             ligand_dict=ligand_dict,
                             fitness_fcns=fitness_fcns,
                             select_on="fx_obs",
                             select_on_folded=True,
                             fitness_kwargs=v,
                             temperature=1)

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
            Fitness(ens=ens,
                             ligand_dict=ligand_dict,
                             fitness_fcns=fitness_fcns,
                             select_on="fx_obs",
                             select_on_folded=True,
                             fitness_kwargs={},
                             temperature=v)

def test_Fitness_fitness():

    # Basic ensemble    
    ens = Ensemble(gas_constant=1)
    ens.add_species(name="test1",
                    observable=True,
                    folded=False,
                    ligand_stoich={"X":1})
    ens.add_species(name="test2",
                    observable=False,
                    folded=False,
                    ligand_stoich={"Y":1})
    
    ligand_dict = {"X":[0,100],"Y":[100,0]}
    fitness_fcns = [ff_off,ff_on]
    select_on = "fx_obs"
    temperature = 1

    fc = Fitness(ens=ens,
                          ligand_dict=ligand_dict,
                          fitness_fcns=fitness_fcns,
                          select_on=select_on,
                          select_on_folded=False,
                          fitness_kwargs={},
                          temperature=temperature)

    mut_energy_array = np.array([0,0]) 
    value = fc.fitness(mut_energy_array=mut_energy_array)
    assert np.array_equal(value,[1,1])

    # Perturb so observable never populated...
    mut_energy_array = np.array([5000,0]) 
    value = fc.fitness(mut_energy_array=mut_energy_array)
    assert np.array_equal(value,[1,0])

    with pytest.raises(ValueError):
        fc = Fitness(ens=ens,
                              ligand_dict=ligand_dict,
                              fitness_fcns=fitness_fcns,
                              select_on="not_right",
                              fitness_kwargs={},
                              temperature=temperature)
        
    # Now set select_on_folded to True. Always unfolded.
    fc = Fitness(ens=ens,
                          ligand_dict=ligand_dict,
                          fitness_fcns=fitness_fcns,
                          select_on=select_on,
                          select_on_folded=True,
                          fitness_kwargs={},
                          temperature=temperature)

    mut_energy_array = np.array([0,0]) 
    value = fc.fitness(mut_energy_array=mut_energy_array)
    assert np.array_equal(value,[0,0])

    # Perturb so observable never populated...
    mut_energy_array = np.array([5000,0]) 
    value = fc.fitness(mut_energy_array=mut_energy_array)
    assert np.array_equal(value,[0,0])

def test_Fitness_to_dict():

    # Basic ensemble    
    ens = Ensemble(gas_constant=1)
    ens.add_species(name="test1",
                    observable=True,
                    folded=False,
                    ligand_stoich={"X":1})
    ens.add_species(name="test2",
                    observable=False,
                    folded=False,
                    ligand_stoich={"Y":1})
    
    ligand_dict = {"X":[0,100],"Y":[100,0]}
    fitness_fcns = [ff_off,ff_on]
    select_on = "fx_obs"
    temperature = 1

    fc = Fitness(ens=ens,
                          ligand_dict=ligand_dict,
                          fitness_fcns=fitness_fcns,
                          select_on=select_on,
                          select_on_folded=False,
                          fitness_kwargs={},
                          temperature=temperature)

    out_dict = fc.to_dict()
    assert np.array_equal(list(out_dict["ligand_dict"].keys()),["X","Y"])
    assert np.array_equal(out_dict["ligand_dict"]["X"],[0,100])
    assert np.array_equal(out_dict["ligand_dict"]["Y"],[100,0])
    assert out_dict["select_on"] == "fx_obs"
    assert out_dict["select_on_folded"] == False
    assert issubclass(type(out_dict["fitness_kwargs"]),dict)
    assert len(out_dict["fitness_kwargs"]) == 0
    assert np.array_equal(out_dict["temperature"],[1,1])
    assert np.array_equal(out_dict["fitness_fcns"],["off","on"])


def test_Fitness_ens():

    # Basic ensemble    
    ens = Ensemble(gas_constant=1)
    ens.add_species(name="test1",
                    observable=True,
                    ligand_stoich={"X":1})
    ens.add_species(name="test2",
                    observable=False,
                    ligand_stoich={"Y":1})
    
    ligand_dict = {"X":[0,100],"Y":[100,0]}
    fitness_fcns = [ff_off,ff_on]
    select_on = "fx_obs"
    temperature = 1

    # make sure attributes are set correctly

    fc = Fitness(ens=ens,
                          ligand_dict=ligand_dict,
                          fitness_fcns=fitness_fcns,
                          select_on=select_on,
                          fitness_kwargs={},
                          temperature=temperature)

    # fc._private_ens should be a copy of ens
    assert fc.ens is ens
    assert fc._private_ens is not ens
    
def test_Fitness_ligand_dict():

    # Basic ensemble    
    ens = Ensemble(gas_constant=1)
    ens.add_species(name="test1",
                    observable=True,
                    folded=False,
                    ligand_stoich={"X":1})
    ens.add_species(name="test2",
                    observable=False,
                    folded=False,
                    ligand_stoich={"Y":1})
    
    ligand_dict = {"X":[0,100],"Y":[100,0]}
    fitness_fcns = [ff_off,ff_on]
    select_on = "fx_obs"
    temperature = 1

    fc = Fitness(ens=ens,
                          ligand_dict=ligand_dict,
                          fitness_fcns=fitness_fcns,
                          select_on=select_on,
                          select_on_folded=False,
                          fitness_kwargs={},
                          temperature=temperature)
    assert np.array_equal(list(fc.ligand_dict.keys()),["X","Y"])
    assert np.array_equal(fc.ligand_dict["X"],[0,100])
    assert np.array_equal(fc.ligand_dict["Y"],[100,0])

def test_Fitness_select_on():

    # Basic ensemble    
    ens = Ensemble(gas_constant=1)
    ens.add_species(name="test1",
                    observable=True,
                    folded=False,
                    ligand_stoich={"X":1})
    ens.add_species(name="test2",
                    observable=False,
                    folded=False,
                    ligand_stoich={"Y":1})
    
    ligand_dict = {"X":[0,100],"Y":[100,0]}
    fitness_fcns = [ff_off,ff_on]
    select_on = "fx_obs"
    temperature = 1

    fc = Fitness(ens=ens,
                          ligand_dict=ligand_dict,
                          fitness_fcns=fitness_fcns,
                          select_on=select_on,
                          select_on_folded=False,
                          fitness_kwargs={},
                          temperature=temperature)
    assert fc.select_on == "fx_obs"

def test_Fitness_select_on_folded():

    # Basic ensemble    
    ens = Ensemble(gas_constant=1)
    ens.add_species(name="test1",
                    observable=True,
                    folded=False,
                    ligand_stoich={"X":1})
    ens.add_species(name="test2",
                    observable=False,
                    folded=False,
                    ligand_stoich={"Y":1})
    
    ligand_dict = {"X":[0,100],"Y":[100,0]}
    fitness_fcns = [ff_off,ff_on]
    select_on = "fx_obs"
    temperature = 1

    fc = Fitness(ens=ens,
                          ligand_dict=ligand_dict,
                          fitness_fcns=fitness_fcns,
                          select_on=select_on,
                          select_on_folded=False,
                          fitness_kwargs={},
                          temperature=temperature)
    assert fc.select_on_folded == False


def test_Fitness_fitness_kwargs():

    # Basic ensemble    
    ens = Ensemble(gas_constant=1)
    ens.add_species(name="test1",
                    observable=True,
                    folded=False,
                    ligand_stoich={"X":1})
    ens.add_species(name="test2",
                    observable=False,
                    folded=False,
                    ligand_stoich={"Y":1})
    
    ligand_dict = {"X":[0,100],"Y":[100,0]}
    fitness_fcns = [ff_off,ff_on]
    select_on = "fx_obs"
    temperature = 1

    fc = Fitness(ens=ens,
                ligand_dict=ligand_dict,
                fitness_fcns=fitness_fcns,
                select_on=select_on,
                select_on_folded=False,
                fitness_kwargs={},
                temperature=temperature)

    assert issubclass(type(fc.fitness_kwargs),dict)
    assert len(fc.fitness_kwargs) == 0

    # Test a threshold. 

    # Basic ensemble    
    ens = Ensemble(gas_constant=1)
    ens.add_species(name="test1",
                    observable=True,
                    folded=False,
                    ligand_stoich={"X":1})
    ens.add_species(name="test2",
                    observable=False,
                    folded=False,
                    ligand_stoich={"Y":1})
    
    ligand_dict = {"X":[0,100],"Y":[100,0]}
    fitness_fcns = ["on_above","on_below"]
    select_on = "fx_obs"
    temperature = 1

    # Both should be zero fitness
    fc = Fitness(ens=ens,
                 ligand_dict=ligand_dict,
                 fitness_fcns=fitness_fcns,
                 select_on=select_on,
                 select_on_folded=False,
                 fitness_kwargs={"threshold":0.5},
                 temperature=temperature)
    
    assert np.array_equal(np.zeros(2),fc.fitness(np.zeros(2)))

    # "on_above" should be fit; "on_below" should not
    fc = Fitness(ens=ens,
                 ligand_dict=ligand_dict,
                 fitness_fcns=fitness_fcns,
                 select_on=select_on,
                 select_on_folded=False,
                 fitness_kwargs={"threshold":0.0},
                 temperature=temperature)
    
    assert np.array_equal([1.0,0.0],fc.fitness(np.zeros(2)))

    # "on_below" should be fit; "on_above" should not
    fc = Fitness(ens=ens,
                 ligand_dict=ligand_dict,
                 fitness_fcns=fitness_fcns,
                 select_on=select_on,
                 select_on_folded=False,
                 fitness_kwargs={"threshold":1.0},
                 temperature=temperature)

    assert np.array_equal([0.0,1.0],fc.fitness(np.zeros(2)))


def test_Fitness_temperature():

    # Basic ensemble    
    ens = Ensemble(gas_constant=1)
    ens.add_species(name="test1",
                    observable=True,
                    ligand_stoich={"X":1})
    ens.add_species(name="test2",
                    observable=False,
                    ligand_stoich={"Y":1})
    
    ligand_dict = {"X":[0,100],"Y":[100,0]}
    fitness_fcns = [ff_off,ff_on]
    select_on = "fx_obs"
    temperature = 1

    # make sure attributes are set correctly
    fc = Fitness(ens=ens,
                 ligand_dict=ligand_dict,
                 fitness_fcns=fitness_fcns,
                 select_on=select_on,
                 fitness_kwargs={},
                 temperature=temperature)
    
    assert np.array_equal(fc.temperature,[1.0,1.0])

    fc = Fitness(ens=ens,
                 ligand_dict=ligand_dict,
                 fitness_fcns=fitness_fcns,
                 select_on=select_on,
                 fitness_kwargs={},
                 temperature=20)
    assert np.array_equal(fc.temperature,[20.0,20.0])

    with pytest.raises(ValueError):
        fc = Fitness(ens=ens,
                     ligand_dict=ligand_dict,
                     fitness_fcns=fitness_fcns,
                     select_on=select_on,
                     fitness_kwargs={},
                     temperature=-2)

def test_Fitness_condition_df():

        # Basic ensemble    
    ens = Ensemble(gas_constant=1)
    ens.add_species(name="test1",
                    observable=True,
                    ligand_stoich={"X":1})
    ens.add_species(name="test2",
                    observable=False,
                    ligand_stoich={"Y":1})
    
    ligand_dict = {"X":[0,100],"Y":[100,0]}
    fitness_fcns = [ff_off,ff_on]
    select_on = "fx_obs"
    temperature = 1

    # make sure attributes are set correctly
    fc = Fitness(ens=ens,
                 ligand_dict=ligand_dict,
                 fitness_fcns=fitness_fcns,
                 select_on=select_on,
                 fitness_kwargs={},
                 temperature=temperature)

    out_df = fc.condition_df
    assert issubclass(type(out_df),pd.DataFrame)
    assert len(out_df) == 2
    assert np.array_equal(out_df["ff"],["off","on"])
    assert np.array_equal(out_df["X"],[0,100])
    assert np.array_equal(out_df["Y"],[100,0])
    assert np.array_equal(out_df["temperature"],[1,1])

