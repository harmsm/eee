
from eee import Ensemble

from eee.evolve.fitness import ff_on
from eee.evolve.fitness import ff_off
from eee.evolve.fitness import ff_neutral
from eee.evolve.fitness import fitness_function
from eee.evolve.fitness import FitnessContainer

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


def test_FitnessContainer():

    # this is just a wrapped version of the _fitness_function. Only new test is
    # for select_on. All other tests are covered by test_eee_variables, 
    # test_check_ensembles, and test_standard unit tests. 

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
  
    assert fc.ens is ens
    assert np.array_equal(fc.T,[1.0,1.0])
    
    for k in fc._mu_dict:
        assert np.array_equal(mu_dict[k],fc._mu_dict[k])
    assert fc._fitness_fcns is fitness_fcns
    assert fc._select_on == select_on
    assert fc._select_on_folded == select_on_folded
    assert len(fc._fitness_kwargs) == 0
    assert issubclass(type(fc._fitness_kwargs),dict)
    assert np.array_equal(fc.T,np.ones(2,dtype=float))



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

    assert fc.ens is ens
    

