
from eee import Ensemble

from eee.evolve.fitness import ff_on
from eee.evolve.fitness import ff_off
from eee.evolve.fitness import _fitness_function
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


def test__fitness_function():
    
    # Basic ensemble
    ens = Ensemble(R=1)
    ens.add_species(name="test1",
                    observable=True,
                    mu_stoich={"X":1})
    ens.add_species(name="test2",
                    observable=False,
                    mu_stoich={"Y":1})
    
    mut_energy = {"test1":0,
                  "test2":0}
    mu_dict = {"X":[0,10000],"Y":[10000,0]}

    T = 1
    select_on = "fx_obs"

    # Try different fitness_fcns

    fitness_fcns = [ff_off,ff_on]
    value = _fitness_function(ens=ens,
                              mut_energy=mut_energy,
                              mu_dict=mu_dict,
                              fitness_fcns=fitness_fcns,
                              select_on=select_on,
                              fitness_kwargs={},
                              T=T)
    
    assert np.array_equal(value,[1,1])

    fitness_fcns = [ff_off,ff_off]
    value = _fitness_function(ens=ens,
                              mut_energy=mut_energy,
                              mu_dict=mu_dict,
                              fitness_fcns=fitness_fcns,
                              select_on=select_on,
                              fitness_kwargs={},
                              T=T)
    
    assert np.array_equal(value,[1,0])

    fitness_fcns = [ff_on,ff_off]
    value = _fitness_function(ens=ens,
                              mut_energy=mut_energy,
                              mu_dict=mu_dict,
                              fitness_fcns=fitness_fcns,
                              select_on=select_on,
                              fitness_kwargs={},
                              T=T)
    
    assert np.array_equal(value,[0,0])

    # Check dG_obs select_on
    ens = Ensemble(R=1)
    ens.add_species(name="test1",
                    observable=True,
                    mu_stoich={"X":1})
    ens.add_species(name="test2",
                    observable=False,
                    mu_stoich={"Y":1})
    
    mut_energy = {"test1":0,
                  "test2":0}
    mu_dict = {"X":[0,1],"Y":[1,0]}

    T = 1
    select_on = "dG_obs"

    fitness_fcns = [ff_on,ff_off]
    value = _fitness_function(ens=ens,
                              mut_energy=mut_energy,
                              mu_dict=mu_dict,
                              fitness_fcns=fitness_fcns,
                              select_on=select_on,
                              fitness_kwargs={},
                              T=T)
    assert np.array_equal(value,[1,2])

def test_fitness_function(variable_types):
    
    # this is just a wrapped version of the _fitness_function. Only new test is
    # for select_on. All other tests are covered by test_eee_variables, 
    # test_check_ensembles, and test_standard unit tests. 

    ens = Ensemble(R=1)
    ens.add_species(name="test1",
                    observable=True,
                    mu_stoich={"X":1})
    ens.add_species(name="test2",
                    observable=False,
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
                            T=T)    


def test_FitnessContainer():
    pass