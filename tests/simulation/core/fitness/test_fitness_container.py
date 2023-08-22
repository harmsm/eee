
from eee import Ensemble

from eee.simulation.core.fitness.fitness import Fitness
from eee.simulation.core.fitness.ff import ff_on
from eee.simulation.core.fitness.ff import ff_off

import numpy as np

import pytest

def test_Fitness(ens_test_data,variable_types):

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
    fc = Fitness(ens=ens,
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
    assert fc._fitness_fcns[0] == fitness_fcns[0]
    assert fc._fitness_fcns[1] == fitness_fcns[1]
    assert fc._select_on == select_on
    assert fc._select_on_folded == select_on_folded
    assert len(fc._fitness_kwargs) == 0
    assert issubclass(type(fc._fitness_kwargs),dict)
    assert np.array_equal(fc.T,np.ones(2,dtype=float))

    ## Check variable type checking

    ens = ens_test_data["ens"]
    mu_dict = ens_test_data["mu_dict"]
    fitness_fcns = ens_test_data["fitness_fcns"]

    # ens
    for v in variable_types["everything"]:
        print(v,type(v),flush=True)    
        with pytest.raises(ValueError):
            Fitness(ens=v,
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
            Fitness(ens=ens,
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
            Fitness(ens=ens,
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
            Fitness(ens=ens,
                             mu_dict=mu_dict,
                             fitness_fcns=fitness_fcns,
                             select_on=v,
                             select_on_folded=True,
                             fitness_kwargs={},
                             T=1)
            
    Fitness(ens=ens,
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
            Fitness(ens=ens,
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
            Fitness(ens=ens,
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
            Fitness(ens=ens,
                             mu_dict=mu_dict,
                             fitness_fcns=fitness_fcns,
                             select_on="fx_obs",
                             select_on_folded=True,
                             fitness_kwargs={},
                             T=v)

def test_Fitness_fitness():

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

    fc = Fitness(ens=ens,
                          mu_dict=mu_dict,
                          fitness_fcns=fitness_fcns,
                          select_on=select_on,
                          select_on_folded=False,
                          fitness_kwargs={},
                          T=T)

    mut_energy_array = np.array([0,0]) 
    value = fc.fitness(mut_energy_array=mut_energy_array)
    assert np.array_equal(value,[1,1])

    # Perturb so observable never populated...
    mut_energy_array = np.array([5000,0]) 
    value = fc.fitness(mut_energy_array=mut_energy_array)
    assert np.array_equal(value,[1,0])

    with pytest.raises(ValueError):
        fc = Fitness(ens=ens,
                              mu_dict=mu_dict,
                              fitness_fcns=fitness_fcns,
                              select_on="not_right",
                              fitness_kwargs={},
                              T=T)
        
    # Now set select_on_folded to True. Always unfolded.
    fc = Fitness(ens=ens,
                          mu_dict=mu_dict,
                          fitness_fcns=fitness_fcns,
                          select_on=select_on,
                          select_on_folded=True,
                          fitness_kwargs={},
                          T=T)

    mut_energy_array = np.array([0,0]) 
    value = fc.fitness(mut_energy_array=mut_energy_array)
    assert np.array_equal(value,[0,0])

    # Perturb so observable never populated...
    mut_energy_array = np.array([5000,0]) 
    value = fc.fitness(mut_energy_array=mut_energy_array)
    assert np.array_equal(value,[0,0])

def test_Fitness_to_dict():

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

    fc = Fitness(ens=ens,
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


def test_Fitness_ens():

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

    fc = Fitness(ens=ens,
                          mu_dict=mu_dict,
                          fitness_fcns=fitness_fcns,
                          select_on=select_on,
                          fitness_kwargs={},
                          T=T)

    # fc._private_ens should be a copy of ens
    assert fc.ens is ens
    assert fc._private_ens is not ens
    
def test_Fitness_mu_dict():

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

    fc = Fitness(ens=ens,
                          mu_dict=mu_dict,
                          fitness_fcns=fitness_fcns,
                          select_on=select_on,
                          select_on_folded=False,
                          fitness_kwargs={},
                          T=T)
    assert np.array_equal(list(fc.mu_dict.keys()),["X","Y"])
    assert np.array_equal(fc.mu_dict["X"],[0,100])
    assert np.array_equal(fc.mu_dict["Y"],[100,0])

def test_Fitness_select_on():

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

    fc = Fitness(ens=ens,
                          mu_dict=mu_dict,
                          fitness_fcns=fitness_fcns,
                          select_on=select_on,
                          select_on_folded=False,
                          fitness_kwargs={},
                          T=T)
    assert fc.select_on == "fx_obs"

def test_Fitness_select_on_folded():

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

    fc = Fitness(ens=ens,
                          mu_dict=mu_dict,
                          fitness_fcns=fitness_fcns,
                          select_on=select_on,
                          select_on_folded=False,
                          fitness_kwargs={},
                          T=T)
    assert fc.select_on_folded == False


def test_Fitness_fitness_kwargs():

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

    fc = Fitness(ens=ens,
                          mu_dict=mu_dict,
                          fitness_fcns=fitness_fcns,
                          select_on=select_on,
                          select_on_folded=False,
                          fitness_kwargs={},
                          T=T)

    assert issubclass(type(fc.fitness_kwargs),dict)
    assert len(fc.fitness_kwargs) == 0

def test_Fitness_T():

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

    fc = Fitness(ens=ens,
                          mu_dict=mu_dict,
                          fitness_fcns=fitness_fcns,
                          select_on=select_on,
                          fitness_kwargs={},
                          T=T)
    
    assert np.array_equal(fc.T,[1.0,1.0])

    fc = Fitness(ens=ens,
                          mu_dict=mu_dict,
                          fitness_fcns=fitness_fcns,
                          select_on=select_on,
                          fitness_kwargs={},
                          T=20)
    assert np.array_equal(fc.T,[20.0,20.0])

    with pytest.raises(ValueError):
        fc = Fitness(ens=ens,
                            mu_dict=mu_dict,
                            fitness_fcns=fitness_fcns,
                            select_on=select_on,
                            fitness_kwargs={},
                            T=-2)


