
from eee import Ensemble

from eee.evolve.fitness import ff_on
from eee.evolve.fitness import ff_off
from eee.evolve.fitness import _fitness_function
from eee.evolve.fitness import fitness_function
from eee.evolve.fitness import FitnessContainer

import numpy as np

def test_ff_on():

    assert ff_on(1) == 1
    assert ff_on(0) == 0


def test_ff_off():

    assert ff_off(1) == 0
    assert ff_off(0) == 1


def test__fitness_function():
    
    # One observable, one not
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

    select_on = "fx_obs"
    fitness_fcns = [ff_off,ff_on]

    T = 1

    value = _fitness_function(ens=ens,
                              mut_energy=mut_energy,
                              mu_dict=mu_dict,
                              fitness_fcns=fitness_fcns,
                              select_on=select_on,
                              fitness_kwargs={},
                              T=T)
    
    assert np.array_equal(value,[1,1])


def test_fitness_function():
    pass

def test_FitnessContainer():
    pass