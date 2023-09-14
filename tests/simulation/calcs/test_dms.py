import pytest

from eee.simulation.calcs import DeepMutationalScan
from eee.io import read_json

import os

def test_DeepMutationalScan(ens_test_data):
    
    ens = ens_test_data["ens"]
    ddg_df = ens_test_data["ddg_df"]
    conditions = ens_test_data["conditions"]

    dms = DeepMutationalScan(ens=ens,
                             ddg_df=ddg_df,
                             conditions=conditions,
                             seed=None)
    
    assert dms.calc_type == "dms"
    
def test_DeepMutationalScan_run(ens_test_data,tmpdir):

    # This class directly wraps engine.exhaustive. Heavy testing there. 
    # Basically make sure it is working here. 

    current_dir = os.getcwd()
    os.chdir(tmpdir)

    ens = ens_test_data["ens"]
    ddg_df = ens_test_data["ddg_df"]
    conditions = ens_test_data["conditions"]

    dms = DeepMutationalScan(ens=ens,
                             ddg_df=ddg_df,
                             conditions=conditions,
                             seed=None)

    dms.run(output_directory="test",
            max_depth=1,
            output_file="yo.csv")
    
    assert os.path.exists(os.path.join("test","yo.csv"))
    

    os.chdir('test')
    _, kwargs = read_json('simulation.json')
    assert kwargs["max_depth"] == 1
    assert kwargs["output_file"] == "yo.csv"

    os.chdir("..")

 
    os.chdir(current_dir)