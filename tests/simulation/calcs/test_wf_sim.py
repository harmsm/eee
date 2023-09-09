import pytest

from eee.simulation.calcs import WrightFisherSimulation
from eee.io import read_json

import pandas as pd

import os

def test_WrightFisherSimulation(ens_test_data):
    
    ens = ens_test_data["ens"]
    ddg_df = ens_test_data["ddg_df"]
    ligand_dict = ens_test_data["ligand_dict"]
    fitness_fcns = ens_test_data["fitness_fcns"]

    wf = WrightFisherSimulation(ens=ens,
                                ddg_df=ddg_df,
                                ligand_dict=ligand_dict,
                                fitness_fcns=fitness_fcns,
                                select_on="fx_obs",
                                fitness_kwargs={},
                                T=1,
                                seed=None)
    
    assert wf.calc_type == "wf_sim"
    
def test_WrightFisherSimulation_run(ens_test_data,tmpdir):

    current_dir = os.getcwd()
    os.chdir(tmpdir)

    ens = ens_test_data["ens"]
    ddg_df = ens_test_data["ddg_df"]
    ligand_dict = ens_test_data["ligand_dict"]
    fitness_fcns = ens_test_data["fitness_fcns"]
    
    wf = WrightFisherSimulation(ens=ens,
                                ddg_df=ddg_df,
                                ligand_dict=ligand_dict,
                                fitness_fcns=fitness_fcns,
                                select_on="fx_obs",
                                fitness_kwargs={},
                                T=1,
                                seed=None)

    wf.run(output_directory="test",
           population_size=100,
           mutation_rate=0.01,
           num_generations=100,
           write_prefix="eee_sim",
           write_frequency=1000)
    
    assert os.path.exists(os.path.join("test","ddg.csv"))
    assert os.path.exists(os.path.join("test","simulation.json"))
    assert os.path.exists(os.path.join("test","eee_sim_genotypes.csv"))
    assert os.path.exists(os.path.join("test","eee_sim_generations_0.pickle"))
 
    os.chdir('test')
    _, kwargs = read_json('simulation.json')
    assert kwargs["population_size"] == 100
    assert kwargs["mutation_rate"] == 0.01
    assert kwargs["num_generations"] == 100
    assert kwargs["write_prefix"] == "eee_sim"
    assert kwargs["write_frequency"] == 1000

    os.chdir("..")

    os.chdir(current_dir)