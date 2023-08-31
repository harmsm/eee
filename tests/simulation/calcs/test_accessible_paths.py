import pytest

from eee.simulation.calcs.accessible_paths import AcessiblePaths
from eee.io import read_json

import os

def test_AcessiblePaths(ens_test_data):
    
    ens = ens_test_data["ens"]
    ddg_df = ens_test_data["ddg_df"]
    mu_dict = ens_test_data["mu_dict"]
    fitness_fcns = ens_test_data["fitness_fcns"]

    ap = AcessiblePaths(ens=ens,
                        ddg_df=ddg_df,
                        mu_dict=mu_dict,
                        fitness_fcns=fitness_fcns,
                        select_on="fx_obs",
                        fitness_kwargs={},
                        T=1,
                        seed=None)
    
    assert ap.calc_type == "accessible_paths"
    
def test_AcessiblePaths_run(ens_test_data,tmpdir):

    # This class directly wraps engine.pathfinder. Heavy testing there. 
    # Basically make sure it is working here. 

    current_dir = os.getcwd()
    os.chdir(tmpdir)

    ens = ens_test_data["ens"]
    ddg_df = ens_test_data["ddg_df"]
    mu_dict = ens_test_data["mu_dict"]
    fitness_fcns = ens_test_data["fitness_fcns"]
    
    ap = AcessiblePaths(ens=ens,
                        ddg_df=ddg_df,
                        mu_dict=mu_dict,
                        fitness_fcns=fitness_fcns,
                        select_on="fx_obs",
                        fitness_kwargs={},
                        T=1,
                        seed=None)

    ap.run(output_directory="test",
           max_depth=2,
           allow_neutral=True,
           find_all_paths=True,
           output_file="yo.csv")
    
    assert os.path.exists(os.path.join("test","yo.csv"))
    
    os.chdir('test')
    _, kwargs = read_json('simulation.json')
    assert kwargs["max_depth"] == 2
    assert kwargs["allow_neutral"] == True
    assert kwargs["find_all_paths"] == True
    assert kwargs["output_file"] == "yo.csv"

    os.chdir("..")

 
    os.chdir(current_dir)