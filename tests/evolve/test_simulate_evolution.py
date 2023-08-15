import pytest

from eee.evolve.simulate_evolution import simulate_evolution

def test_simulate_evolution(variable_types,ens_test_data,ens_with_fitness):

    # This wraps wright_fisher, which is directly tested. Main goal of this 
    # test is to make sure it runs without error and to validate the error 
    # checking on the input variables. 

    ens = ens_test_data["ens"]
    ddg_df = ens_test_data["ddg_df"]
    mu_dict = ens_test_data["mu_dict"]
    fitness_fcns = ens_test_data["fitness_fcns"]
    
    simulate_evolution(ens=ens,
                       ddg_df=ddg_df,
                       mu_dict=mu_dict,
                       fitness_fcns=fitness_fcns,
                       select_on="fx_obs",
                       fitness_kwargs={},
                       T=1,
                       population_size=100,
                       mutation_rate=0.01,
                       num_generations=100,
                       write_prefix=None,
                       write_frequency=5)
    
    # ens
    for v in variable_types["everything"]:
        print(v,type(v))
        with pytest.raises(ValueError):
            simulate_evolution(ens=v,
                               ddg_df=ddg_df,
                               mu_dict=mu_dict,
                               fitness_fcns=fitness_fcns,
                               select_on="fx_obs",
                               fitness_kwargs={},
                               T=1,
                               population_size=100,
                               mutation_rate=0.01,
                               num_generations=100,
                               write_prefix=None,
                               write_frequency=5)
            
    # ddg_df
    for v in variable_types["everything"]:
        print(v,type(v))
        with pytest.raises(ValueError):
            simulate_evolution(ens=ens,
                               ddg_df=v,
                               mu_dict=mu_dict,
                               fitness_fcns=fitness_fcns,
                               select_on="fx_obs",
                               fitness_kwargs={},
                               T=1,
                               population_size=100,
                               mutation_rate=0.01,
                               num_generations=100,
                               write_prefix=None,
                               write_frequency=5)

    # mu_dict
    for v in variable_types["everything"]:
        print(v,type(v))
        with pytest.raises(ValueError):
            simulate_evolution(ens=ens,
                               ddg_df=ddg_df,
                               mu_dict=v,
                               fitness_fcns=fitness_fcns,
                               select_on="fx_obs",
                               fitness_kwargs={},
                               T=1,
                               population_size=100,
                               mutation_rate=0.01,
                               num_generations=100,
                               write_prefix=None,
                               write_frequency=5)

