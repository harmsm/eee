import pytest

from eee.simulation.calcs import DeepMutationalScan

import os

def test_DeepMutationalScan(ens_test_data):
    
    ens = ens_test_data["ens"]
    ddg_df = ens_test_data["ddg_df"]
    mu_dict = ens_test_data["mu_dict"]
    fitness_fcns = ens_test_data["fitness_fcns"]

    dms = DeepMutationalScan(ens=ens,
                             ddg_df=ddg_df,
                             mu_dict=mu_dict,
                             fitness_fcns=fitness_fcns,
                             select_on="fx_obs",
                             fitness_kwargs={},
                             T=1,
                             seed=None)
    
    assert dms.calc_type == "dms"
    
def test_DeepMutationalScan_run(ens_test_data,tmpdir):

    # This class directly wraps engine.exhaustive. Heavy testing there. 
    # Basically make sure it is working here. 

    current_dir = os.getcwd()
    os.chdir(tmpdir)

    ens = ens_test_data["ens"]
    ddg_df = ens_test_data["ddg_df"]
    mu_dict = ens_test_data["mu_dict"]
    fitness_fcns = ens_test_data["fitness_fcns"]
    
    dms = DeepMutationalScan(ens=ens,
                             ddg_df=ddg_df,
                             mu_dict=mu_dict,
                             fitness_fcns=fitness_fcns,
                             select_on="fx_obs",
                             fitness_kwargs={},
                             T=1,
                             seed=None)

    dms.run(output_directory="test",
            max_depth=1,
            output_file="yo.csv")
    
    assert os.path.exists(os.path.join("test","yo.csv"))
    
 
    os.chdir(current_dir)