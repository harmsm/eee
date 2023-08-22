
from eee import Ensemble

from eee.simulation.analysis import fitness
from eee.simulation.core.fitness.ff import ff_on
from eee.simulation.core.fitness.ff import ff_off

import numpy as np
import pandas as pd

import pytest


def test_fitness(ens_with_fitness):
    
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
    value = fitness(ens=ens,
                    mut_energy=mut_energy,
                    mu_dict=mu_dict,
                    fitness_fcns=fitness_fcns,
                    select_on=select_on,
                    select_on_folded=False,
                    fitness_kwargs={},
                    T=T)
    
    assert np.array_equal(value["fitness"],[1,1])
    assert issubclass(type(value),pd.DataFrame)

    # Try different fitness_fcns
    fitness_fcns = ["off","on"]
    value = fitness(ens=ens,
                    mut_energy=mut_energy,
                    mu_dict=mu_dict,
                    fitness_fcns=fitness_fcns,
                    select_on=select_on,
                    select_on_folded=False,
                    fitness_kwargs={},
                    T=T)
    
    assert np.array_equal(value["fitness"],[1,1])

    # Now select on folded. This will be folded under first condition (test2 
    # favored) and not folded under second condition (test1 favored). 
    value = fitness(ens=ens,
                    mut_energy=mut_energy,
                    mu_dict=mu_dict,
                    fitness_fcns=fitness_fcns,
                    select_on=select_on,
                    select_on_folded=True,
                    fitness_kwargs={},
                    T=T)
    
    assert np.array_equal(value["fitness"],[1,0])


    fitness_fcns = [ff_off,ff_off]
    value = fitness(ens=ens,
                    mut_energy=mut_energy,
                    mu_dict=mu_dict,
                    fitness_fcns=fitness_fcns,
                    select_on=select_on,
                    select_on_folded=False,
                    fitness_kwargs={},
                    T=T)
    
    assert np.array_equal(value["fitness"],[1,0])

    fitness_fcns = [ff_on,ff_off]
    value = fitness(ens=ens,
                    mut_energy=mut_energy,
                    mu_dict=mu_dict,
                    fitness_fcns=fitness_fcns,
                    select_on=select_on,
                    select_on_folded=False,
                    fitness_kwargs={},
                    T=T)
    
    assert np.array_equal(value["fitness"],[0,0])

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
    value = fitness(ens=ens,
                    mut_energy=mut_energy,
                    mu_dict=mu_dict,
                    fitness_fcns=fitness_fcns,
                    select_on=select_on,
                    select_on_folded=False,
                    fitness_kwargs={},
                    T=T)
    assert np.array_equal(value["fitness"],[1,2])

    # Now select on folded. Will be entirely unfolded --> 0,0
    value = fitness(ens=ens,
                    mut_energy=mut_energy,
                    mu_dict=mu_dict,
                    fitness_fcns=fitness_fcns,
                    select_on=select_on,
                    select_on_folded=True,
                    fitness_kwargs={},
                    T=T)
    
    assert np.array_equal(value["fitness"],[0,0])


    ens = ens_with_fitness["ens"]
    mu_dict = ens_with_fitness["mu_dict"]

    value = fitness(ens=ens,
                    mut_energy={},
                    mu_dict=mu_dict,
                    fitness_fcns=[ff_on,ff_off],
                    select_on="fx_obs",
                    fitness_kwargs={},
                    select_on_folded=False,
                    T=1)
    assert np.array_equal(np.round(value["fitness"],2),[0.54,1.00])

    value = fitness(ens=ens,
                    mut_energy={"s1":-1.667,"s2":3.333},
                    mu_dict=mu_dict,
                    fitness_fcns=[ff_on,ff_off],
                    select_on="fx_obs",
                    fitness_kwargs={},
                    select_on_folded=False,
                    T=1)
    assert np.array_equal(np.round(value["fitness"],2),[0.99,0.82])
    
    value = fitness(ens=ens,
                    mut_energy={"s1":0.167,"s2":-16.667},
                    mu_dict=mu_dict,
                    fitness_fcns=[ff_on,ff_off],
                    select_on="fx_obs",
                    fitness_kwargs={},
                    select_on_folded=False,
                    T=1)
    assert np.array_equal(np.round(value["fitness"],2),[0.00,1.00])

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

    value = fitness(ens=ens,
                    mut_energy=mut_energy,
                    mu_dict=mu_dict,
                    fitness_fcns=fitness_fcns,
                    select_on=select_on,
                    select_on_folded=False,
                    fitness_kwargs={},
                    T=T)    
    
    assert np.array_equal(value["fitness"],[1,1])

    with pytest.raises(ValueError):
        value = fitness(ens=ens,
                        mut_energy=mut_energy,
                        mu_dict=mu_dict,
                        fitness_fcns=fitness_fcns,
                        select_on="not_right",
                        fitness_kwargs={},
                        select_on_folded=False,
                        T=T)    
