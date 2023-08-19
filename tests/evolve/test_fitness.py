
from eee import Ensemble

from eee.evolve.fitness import ff_on
from eee.evolve.fitness import ff_off
from eee.evolve.fitness import ff_neutral
from eee.evolve.fitness import get_fitness_function
from eee.evolve.fitness import fitness_function
from eee.evolve.fitness import FitnessContainer
from eee.evolve.fitness import FITNESS_FUNCTION_TO_STR
from eee.evolve.fitness import FITNESS_STR_TO_FUNCTION

import numpy as np

import pytest

def test_ff_on():
    assert ff_on(1) == 1
    assert ff_on(0) == 0

def test_ff_off():
    assert ff_off(1) == 0
    assert ff_off(0) == 1

def test_ff_neutral():
    assert ff_neutral(0) == 1
    assert ff_neutral(1) == 1

def test_get_fitness_function(variable_types):

    assert get_fitness_function("on") is ff_on
    assert get_fitness_function("off") is ff_off
    assert get_fitness_function("neutral") is ff_neutral

    for v in variable_types:
        print(v,type(v),flush=True)
        with pytest.raises(ValueError):
            get_fitness_function(v)

    assert FITNESS_STR_TO_FUNCTION["on"] == ff_on
    assert FITNESS_STR_TO_FUNCTION["off"] == ff_off
    assert FITNESS_STR_TO_FUNCTION["neutral"] == ff_neutral

    assert FITNESS_FUNCTION_TO_STR[ff_on] == "on"
    assert FITNESS_FUNCTION_TO_STR[ff_off] == "off"
    assert FITNESS_FUNCTION_TO_STR[ff_neutral] == "neutral"

def test_fitness_function(ens_with_fitness):
    
    # Basic ensemble
    ens = Ensemble(R=1)
    ens.add_species(name="test1",
                    observable=True,
                    folded=False,
                    mu_stoich={"X":1})
    ens.add_species(name="test2",
                    observable=False,
                    folded=True,
                    mu_stoich={"Y":1})
    
    mut_energy = {"test1":0,
                  "test2":0}
    mu_dict = {"X":[0,10000],"Y":[10000,0]}

    T = 1
    select_on = "fx_obs"

    # Try different fitness_fcns
    fitness_fcns = [ff_off,ff_on]
    value = fitness_function(ens=ens,
                             mut_energy=mut_energy,
                             mu_dict=mu_dict,
                             fitness_fcns=fitness_fcns,
                             select_on=select_on,
                             select_on_folded=False,
                             fitness_kwargs={},
                             T=T)
    
    assert np.array_equal(value,[1,1])

    # Now select on folded. This will be folded under first condition (test2 
    # favored) and not folded under second condition (test1 favored). 
    value = fitness_function(ens=ens,
                             mut_energy=mut_energy,
                             mu_dict=mu_dict,
                             fitness_fcns=fitness_fcns,
                             select_on=select_on,
                             select_on_folded=True,
                             fitness_kwargs={},
                             T=T)
    
    assert np.array_equal(value,[1,0])


    fitness_fcns = [ff_off,ff_off]
    value = fitness_function(ens=ens,
                             mut_energy=mut_energy,
                             mu_dict=mu_dict,
                             fitness_fcns=fitness_fcns,
                             select_on=select_on,
                             select_on_folded=False,
                             fitness_kwargs={},
                             T=T)
    
    assert np.array_equal(value,[1,0])

    fitness_fcns = [ff_on,ff_off]
    value = fitness_function(ens=ens,
                             mut_energy=mut_energy,
                             mu_dict=mu_dict,
                             fitness_fcns=fitness_fcns,
                             select_on=select_on,
                             select_on_folded=False,
                             fitness_kwargs={},
                             T=T)
    
    assert np.array_equal(value,[0,0])

    # Check dG_obs select_on
    ens = Ensemble(R=1)
    ens.add_species(name="test1",
                    observable=True,
                    folded=False,
                    mu_stoich={"X":1})
    ens.add_species(name="test2",
                    observable=False,
                    folded=False,
                    mu_stoich={"Y":1})
    
    mut_energy = {"test1":0,
                  "test2":0}
    mu_dict = {"X":[0,1],"Y":[1,0]}

    T = 1
    select_on = "dG_obs"

    fitness_fcns = [ff_on,ff_off]
    value = fitness_function(ens=ens,
                             mut_energy=mut_energy,
                             mu_dict=mu_dict,
                             fitness_fcns=fitness_fcns,
                             select_on=select_on,
                             select_on_folded=False,
                             fitness_kwargs={},
                             T=T)
    assert np.array_equal(value,[1,2])

    # Now select on folded. Will be entirely unfolded --> 0,0
    value = fitness_function(ens=ens,
                             mut_energy=mut_energy,
                             mu_dict=mu_dict,
                             fitness_fcns=fitness_fcns,
                             select_on=select_on,
                             select_on_folded=True,
                             fitness_kwargs={},
                             T=T)
    
    assert np.array_equal(value,[0,0])


    ens = ens_with_fitness["ens"]
    mu_dict = ens_with_fitness["mu_dict"]

    value = fitness_function(ens=ens,
                             mut_energy={},
                             mu_dict=mu_dict,
                             fitness_fcns=[ff_on,ff_off],
                             select_on="fx_obs",
                             fitness_kwargs={},
                             select_on_folded=False,
                             T=1)
    assert np.array_equal(np.round(value,2),[0.54,1.00])

    value = fitness_function(ens=ens,
                             mut_energy={"s1":-1.667,"s2":3.333},
                             mu_dict=mu_dict,
                             fitness_fcns=[ff_on,ff_off],
                             select_on="fx_obs",
                             fitness_kwargs={},
                             select_on_folded=False,
                             T=1)
    assert np.array_equal(np.round(value,2),[0.99,0.82])
    
    value = fitness_function(ens=ens,
                             mut_energy={"s1":0.167,"s2":-16.667},
                             mu_dict=mu_dict,
                             fitness_fcns=[ff_on,ff_off],
                             select_on="fx_obs",
                             fitness_kwargs={},
                             select_on_folded=False,
                             T=1)
    assert np.array_equal(np.round(value,2),[0.00,1.00])

    ens = Ensemble(R=1)
    ens.add_species(name="test1",
                    observable=True,
                    folded=False,
                    mu_stoich={"X":1})
    ens.add_species(name="test2",
                    observable=False,
                    folded=True,
                    mu_stoich={"Y":1})
    
    mut_energy = {"test1":0,
                  "test2":0}
    mu_dict = {"X":[0,10000],"Y":[10000,0]}
    fitness_fcns = [ff_off,ff_on]

    select_on = "fx_obs"
    T = 1

    value = fitness_function(ens=ens,
                             mut_energy=mut_energy,
                             mu_dict=mu_dict,
                             fitness_fcns=fitness_fcns,
                             select_on=select_on,
                             select_on_folded=False,
                             fitness_kwargs={},
                             T=T)    
    
    assert np.array_equal(value,[1,1])

    with pytest.raises(ValueError):
        value = fitness_function(ens=ens,
                                 mut_energy=mut_energy,
                                 mu_dict=mu_dict,
                                 fitness_fcns=fitness_fcns,
                                 select_on="not_right",
                                 fitness_kwargs={},
                                 select_on_folded=False,
                                 T=T)    


def test_FitnessContainer(ens_test_data,variable_types):

    # Basic ensemble    
    ens = Ensemble(R=1)
    ens.add_species(name="test1",
                    observable=True,
                    folded=False,
                    mu_stoich={"X":1})
    ens.add_species(name="test2",
                    observable=False,
                    folded=True,
                    mu_stoich={"Y":1})
    
    mu_dict = {"X":[0,10000],"Y":[10000,0]}
    fitness_fcns = [ff_off,ff_on]
    select_on = "fx_obs"
    select_on_folded = True
    T = 1

    # make sure attributes are set correctly
    fc = FitnessContainer(ens=ens,
                          mu_dict=mu_dict,
                          fitness_fcns=fitness_fcns,
                          select_on=select_on,
                          select_on_folded=select_on_folded,
                          fitness_kwargs={},
                          T=T)
  
    # fc._private_ens should be a copy of ens
    assert fc.ens is ens
    assert fc._private_ens is not ens

    assert np.array_equal(fc.T,[1.0,1.0])
    
    for k in fc._mu_dict:
        assert np.array_equal(mu_dict[k],fc._mu_dict[k])
    assert fc._fitness_fcns is fitness_fcns
    assert fc._select_on == select_on
    assert fc._select_on_folded == select_on_folded
    assert len(fc._fitness_kwargs) == 0
    assert issubclass(type(fc._fitness_kwargs),dict)
    assert np.array_equal(fc.T,np.ones(2,dtype=float))

    ## Check variable type checking

    ens = ens_test_data["ens"]
    ddg_df = ens_test_data["ddg_df"]
    mu_dict = ens_test_data["mu_dict"]
    fitness_fcns = ens_test_data["fitness_fcns"]

    # ens
    for v in variable_types["everything"]:
        print(v,type(v),flush=True)    
        with pytest.raises(ValueError):
            FitnessContainer(ens=v,
                             mu_dict=mu_dict,
                             fitness_fcns=fitness_fcns,
                             select_on="fx_obs",
                             select_on_folded=True,
                             fitness_kwargs={},
                             T=1)
        
    # mu_dict
    for v in variable_types["everything"]:
        print(v,type(v),flush=True)
        with pytest.raises(ValueError):
            FitnessContainer(ens=ens,
                             mu_dict=v,
                             fitness_fcns=fitness_fcns,
                             select_on="fx_obs",
                             select_on_folded=True,
                             fitness_kwargs={},
                             T=1)

    # fitness_fcns
    for v in variable_types["everything"]:
        print(v,type(v),flush=True)
        with pytest.raises(ValueError):
            FitnessContainer(ens=ens,
                             mu_dict=mu_dict,
                             fitness_fcns=v,
                             select_on="fx_obs",
                             select_on_folded=True,
                             fitness_kwargs={},
                             T=1)
            
    # select_on
    for v in variable_types["everything"]:
        print(v,type(v),flush=True)
        with pytest.raises(ValueError):
            FitnessContainer(ens=ens,
                             mu_dict=mu_dict,
                             fitness_fcns=fitness_fcns,
                             select_on=v,
                             select_on_folded=True,
                             fitness_kwargs={},
                             T=1)
            
    FitnessContainer(ens=ens,
                     mu_dict=mu_dict,
                     fitness_fcns=fitness_fcns,
                     select_on="dG_obs",
                     select_on_folded=True,
                     fitness_kwargs={},
                     T=1)
    
    # select_on_folded
    for v in variable_types["not_bools"]:
        print(v,type(v),flush=True)
        with pytest.raises(ValueError):
            FitnessContainer(ens=ens,
                             mu_dict=mu_dict,
                             fitness_fcns=fitness_fcns,
                             select_on="fx_obs",
                             select_on_folded=v,
                             fitness_kwargs={},
                             T=1)

    # fitness_kwargs
    for v in variable_types["everything"]:
        if issubclass(type(v),dict):
            continue
        if v is None:
            continue

        print(v,type(v),flush=True)
        with pytest.raises(ValueError):
            FitnessContainer(ens=ens,
                             mu_dict=mu_dict,
                             fitness_fcns=fitness_fcns,
                             select_on="fx_obs",
                             select_on_folded=True,
                             fitness_kwargs=v,
                             T=1)

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
            FitnessContainer(ens=ens,
                             mu_dict=mu_dict,
                             fitness_fcns=fitness_fcns,
                             select_on="fx_obs",
                             select_on_folded=True,
                             fitness_kwargs={},
                             T=v)

def test_FitnessContainer_fitness():

    # Basic ensemble    
    ens = Ensemble(R=1)
    ens.add_species(name="test1",
                    observable=True,
                    folded=False,
                    mu_stoich={"X":1})
    ens.add_species(name="test2",
                    observable=False,
                    folded=False,
                    mu_stoich={"Y":1})
    
    mu_dict = {"X":[0,100],"Y":[100,0]}
    fitness_fcns = [ff_off,ff_on]
    select_on = "fx_obs"
    T = 1

    fc = FitnessContainer(ens=ens,
                          mu_dict=mu_dict,
                          fitness_fcns=fitness_fcns,
                          select_on=select_on,
                          select_on_folded=False,
                          fitness_kwargs={},
                          T=T)

    mut_energy_array = np.array([0,0]) 
    assert fc.fitness(mut_energy_array=mut_energy_array) == 1

    # Perturb so observable never populated...
    mut_energy_array = np.array([5000,0]) 
    assert fc.fitness(mut_energy_array=mut_energy_array) == 0

    with pytest.raises(ValueError):
        fc = FitnessContainer(ens=ens,
                              mu_dict=mu_dict,
                              fitness_fcns=fitness_fcns,
                              select_on="not_right",
                              fitness_kwargs={},
                              T=T)
        
    # Now set select_on_folded to True. Always unfolded.
    fc = FitnessContainer(ens=ens,
                          mu_dict=mu_dict,
                          fitness_fcns=fitness_fcns,
                          select_on=select_on,
                          select_on_folded=True,
                          fitness_kwargs={},
                          T=T)

    mut_energy_array = np.array([0,0]) 
    assert fc.fitness(mut_energy_array=mut_energy_array) == 0

    # Perturb so observable never populated...
    mut_energy_array = np.array([5000,0]) 
    assert fc.fitness(mut_energy_array=mut_energy_array) == 0

def test_FitnessContainer_to_dict():

    # Basic ensemble    
    ens = Ensemble(R=1)
    ens.add_species(name="test1",
                    observable=True,
                    folded=False,
                    mu_stoich={"X":1})
    ens.add_species(name="test2",
                    observable=False,
                    folded=False,
                    mu_stoich={"Y":1})
    
    mu_dict = {"X":[0,100],"Y":[100,0]}
    fitness_fcns = [ff_off,ff_on]
    select_on = "fx_obs"
    T = 1

    fc = FitnessContainer(ens=ens,
                          mu_dict=mu_dict,
                          fitness_fcns=fitness_fcns,
                          select_on=select_on,
                          select_on_folded=False,
                          fitness_kwargs={},
                          T=T)

    out_dict = fc.to_dict()
    assert np.array_equal(list(out_dict["mu_dict"].keys()),["X","Y"])
    assert np.array_equal(out_dict["mu_dict"]["X"],[0,100])
    assert np.array_equal(out_dict["mu_dict"]["Y"],[100,0])
    assert out_dict["select_on"] == "fx_obs"
    assert out_dict["select_on_folded"] == False
    assert issubclass(type(out_dict["fitness_kwargs"]),dict)
    assert len(out_dict["fitness_kwargs"]) == 0
    assert np.array_equal(out_dict["T"],[1,1])
    assert np.array_equal(out_dict["fitness_fcns"],["off","on"])


def test_FitnessContainer_ens():

    # Basic ensemble    
    ens = Ensemble(R=1)
    ens.add_species(name="test1",
                    observable=True,
                    mu_stoich={"X":1})
    ens.add_species(name="test2",
                    observable=False,
                    mu_stoich={"Y":1})
    
    mu_dict = {"X":[0,100],"Y":[100,0]}
    fitness_fcns = [ff_off,ff_on]
    select_on = "fx_obs"
    T = 1

    # make sure attributes are set correctly

    fc = FitnessContainer(ens=ens,
                          mu_dict=mu_dict,
                          fitness_fcns=fitness_fcns,
                          select_on=select_on,
                          fitness_kwargs={},
                          T=T)

    # fc._private_ens should be a copy of ens
    assert fc.ens is ens
    assert fc._private_ens is not ens
    
def test_FitnessContainer_mu_dict():

    # Basic ensemble    
    ens = Ensemble(R=1)
    ens.add_species(name="test1",
                    observable=True,
                    folded=False,
                    mu_stoich={"X":1})
    ens.add_species(name="test2",
                    observable=False,
                    folded=False,
                    mu_stoich={"Y":1})
    
    mu_dict = {"X":[0,100],"Y":[100,0]}
    fitness_fcns = [ff_off,ff_on]
    select_on = "fx_obs"
    T = 1

    fc = FitnessContainer(ens=ens,
                          mu_dict=mu_dict,
                          fitness_fcns=fitness_fcns,
                          select_on=select_on,
                          select_on_folded=False,
                          fitness_kwargs={},
                          T=T)
    assert np.array_equal(list(fc.mu_dict.keys()),["X","Y"])
    assert np.array_equal(fc.mu_dict["X"],[0,100])
    assert np.array_equal(fc.mu_dict["Y"],[100,0])

def test_FitnessContainer_select_on():

    # Basic ensemble    
    ens = Ensemble(R=1)
    ens.add_species(name="test1",
                    observable=True,
                    folded=False,
                    mu_stoich={"X":1})
    ens.add_species(name="test2",
                    observable=False,
                    folded=False,
                    mu_stoich={"Y":1})
    
    mu_dict = {"X":[0,100],"Y":[100,0]}
    fitness_fcns = [ff_off,ff_on]
    select_on = "fx_obs"
    T = 1

    fc = FitnessContainer(ens=ens,
                          mu_dict=mu_dict,
                          fitness_fcns=fitness_fcns,
                          select_on=select_on,
                          select_on_folded=False,
                          fitness_kwargs={},
                          T=T)
    assert fc.select_on == "fx_obs"

def test_FitnessContainer_select_on_folded():

    # Basic ensemble    
    ens = Ensemble(R=1)
    ens.add_species(name="test1",
                    observable=True,
                    folded=False,
                    mu_stoich={"X":1})
    ens.add_species(name="test2",
                    observable=False,
                    folded=False,
                    mu_stoich={"Y":1})
    
    mu_dict = {"X":[0,100],"Y":[100,0]}
    fitness_fcns = [ff_off,ff_on]
    select_on = "fx_obs"
    T = 1

    fc = FitnessContainer(ens=ens,
                          mu_dict=mu_dict,
                          fitness_fcns=fitness_fcns,
                          select_on=select_on,
                          select_on_folded=False,
                          fitness_kwargs={},
                          T=T)
    assert fc.select_on_folded == False


def test_FitnessContainer_fitness_kwargs():

    # Basic ensemble    
    ens = Ensemble(R=1)
    ens.add_species(name="test1",
                    observable=True,
                    folded=False,
                    mu_stoich={"X":1})
    ens.add_species(name="test2",
                    observable=False,
                    folded=False,
                    mu_stoich={"Y":1})
    
    mu_dict = {"X":[0,100],"Y":[100,0]}
    fitness_fcns = [ff_off,ff_on]
    select_on = "fx_obs"
    T = 1

    fc = FitnessContainer(ens=ens,
                          mu_dict=mu_dict,
                          fitness_fcns=fitness_fcns,
                          select_on=select_on,
                          select_on_folded=False,
                          fitness_kwargs={},
                          T=T)

    assert issubclass(type(fc.fitness_kwargs),dict)
    assert len(fc.fitness_kwargs) == 0

def test_FitnessContainer_T():

    # Basic ensemble    
    ens = Ensemble(R=1)
    ens.add_species(name="test1",
                    observable=True,
                    mu_stoich={"X":1})
    ens.add_species(name="test2",
                    observable=False,
                    mu_stoich={"Y":1})
    
    mu_dict = {"X":[0,100],"Y":[100,0]}
    fitness_fcns = [ff_off,ff_on]
    select_on = "fx_obs"
    T = 1

    # make sure attributes are set correctly

    fc = FitnessContainer(ens=ens,
                          mu_dict=mu_dict,
                          fitness_fcns=fitness_fcns,
                          select_on=select_on,
                          fitness_kwargs={},
                          T=T)
    
    assert np.array_equal(fc.T,[1.0,1.0])

    fc = FitnessContainer(ens=ens,
                          mu_dict=mu_dict,
                          fitness_fcns=fitness_fcns,
                          select_on=select_on,
                          fitness_kwargs={},
                          T=20)
    assert np.array_equal(fc.T,[20.0,20.0])

    with pytest.raises(ValueError):
        fc = FitnessContainer(ens=ens,
                            mu_dict=mu_dict,
                            fitness_fcns=fitness_fcns,
                            select_on=select_on,
                            fitness_kwargs={},
                            T=-2)


