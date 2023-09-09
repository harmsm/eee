import pytest

from eee.io.read_ensemble import _search_for_key
from eee.io.read_ensemble import _spreadsheet_to_ensemble
from eee.io.read_ensemble import _json_to_ensemble
from eee.io.read_ensemble import read_ensemble

import numpy as np
import pandas as pd

import os
import shutil

def test__search_for_key():

    some_dict = {"test":{"this":{"out":1},
                         "stupid":{1:5},
                         "rocket":{"ship":2}
                        },
                  "x":[1,2,3]}

    out = _search_for_key(some_dict,"test")
    assert np.array_equal(out,["test"])

    out = _search_for_key(some_dict,"this")
    assert np.array_equal(out,["test","this"])

    out = _search_for_key(some_dict,"ship")
    assert np.array_equal(out,["test","rocket","ship"])

    out = _search_for_key(some_dict,"x")
    assert np.array_equal(out,["x"])

    out = _search_for_key(some_dict,"not_a_key")
    assert len(out) == 0
    
def test__spreadsheet_to_ensemble(ensemble_inputs):
    
    df = ensemble_inputs["s1s2_folded.xlsx"]

    # Basic read 
    ens = _spreadsheet_to_ensemble(df=df,
                                   gas_constant=None)
    assert np.array_equal(ens.species,["s1","s2"])

    assert ens._species["s1"]["dG0"] == 0
    assert ens._species["s1"]["ligand_stoich"]["X"] == 1
    assert ens._species["s1"]["ligand_stoich"]["Y"] == 0
    assert ens._species["s1"]["folded"] == True
    assert ens._species["s1"]["observable"] == True
        
    assert ens._species["s2"]["dG0"] == 10
    assert ens._species["s2"]["ligand_stoich"]["X"] == 0
    assert ens._species["s2"]["ligand_stoich"]["Y"] == 1
    assert ens._species["s2"]["folded"] == True
    assert ens._species["s2"]["observable"] == False

    assert ens._gas_constant == 0.001987

    # One change to entry 
    df = ensemble_inputs["s1s2_unfolded.xlsx"]

    ens = _spreadsheet_to_ensemble(df=df,
                                   gas_constant=1.0)
    assert np.array_equal(ens.species,["s1","s2"])

    assert ens._species["s1"]["dG0"] == 0
    assert ens._species["s1"]["ligand_stoich"]["X"] == 1
    assert ens._species["s1"]["ligand_stoich"]["Y"] == 0
    assert ens._species["s1"]["folded"] == True
    assert ens._species["s1"]["observable"] == True
        
    assert ens._species["s2"]["dG0"] == 10
    assert ens._species["s2"]["ligand_stoich"]["X"] == 0
    assert ens._species["s2"]["ligand_stoich"]["Y"] == 1
    assert ens._species["s2"]["folded"] == False
    assert ens._species["s2"]["observable"] == False   
    
    assert ens._gas_constant == 1.0

    # Drop some columns
    df = ensemble_inputs["s1s2_nostoich.xlsx"]

    ens = _spreadsheet_to_ensemble(df=df,
                                   gas_constant=None)
    assert np.array_equal(ens.species,["s1","s2"])

    assert ens._species["s1"]["dG0"] == 0
    assert ens._species["s1"]["ligand_stoich"] == {}
    assert ens._species["s1"]["folded"] == True
    assert ens._species["s1"]["observable"] == True
        
    assert ens._species["s2"]["dG0"] == 10
    assert ens._species["s2"]["ligand_stoich"] == {}
    assert ens._species["s2"]["folded"] == False
    assert ens._species["s2"]["observable"] == False   
    
    # Totally minimal
    df = pd.DataFrame({"name":["s1","s2","s3"]})

    ens = _spreadsheet_to_ensemble(df=df)
    assert ens._species["s1"]["dG0"] == 0
    assert ens._species["s1"]["ligand_stoich"] == {}
    assert ens._species["s1"]["folded"] == True
    assert ens._species["s1"]["observable"] == False
        
    assert ens._species["s2"]["dG0"] == 0
    assert ens._species["s2"]["ligand_stoich"] == {}
    assert ens._species["s2"]["folded"] == True
    assert ens._species["s2"]["observable"] == False  

    assert ens._species["s3"]["dG0"] == 0
    assert ens._species["s3"]["ligand_stoich"] == {}
    assert ens._species["s3"]["folded"] == True
    assert ens._species["s3"]["observable"] == False  

    # Missing something required (name only required, actually)
    df = pd.DataFrame({"dG0":[1,3]})
    with pytest.raises(ValueError):
        ens = _spreadsheet_to_ensemble(df=df)

def test__json_to_ensemble(sim_json,ensemble_inputs,tmpdir):
    
    test_json = sim_json["dms.json"]

    ens = _json_to_ensemble(test_json)

    assert ens._gas_constant == 0.001987

    assert ens._species["hdna"]["dG0"] == 0
    assert ens._species["hdna"]["ligand_stoich"] == {}
    assert ens._species["hdna"]["folded"] == True
    assert ens._species["hdna"]["observable"] == True

    assert ens._species["h"]["dG0"] == 5
    assert ens._species["h"]["ligand_stoich"] == {}
    assert ens._species["h"]["folded"] == True
    assert ens._species["h"]["observable"] == False

    assert ens._species["l2e"]["dG0"] == 5
    assert ens._species["l2e"]["ligand_stoich"] == {"iptg":4}
    assert ens._species["l2e"]["folded"] == True
    assert ens._species["l2e"]["observable"] == False

    assert ens._species["unfolded"]["dG0"] == 10
    assert ens._species["unfolded"]["ligand_stoich"] == {}
    assert ens._species["unfolded"]["folded"] == False
    assert ens._species["unfolded"]["observable"] == False

    test_json = sim_json["lac.json"]

    ens = _json_to_ensemble(test_json)

    # Test gas constant reading

    assert ens._gas_constant == 0.008314

    assert ens._species["hdna"]["dG0"] == 0
    assert ens._species["hdna"]["ligand_stoich"] == {}
    assert ens._species["hdna"]["folded"] == True
    assert ens._species["hdna"]["observable"] == True

    assert ens._species["h"]["dG0"] == 5
    assert ens._species["h"]["ligand_stoich"] == {}
    assert ens._species["h"]["folded"] == True
    assert ens._species["h"]["observable"] == False

    assert ens._species["l2e"]["dG0"] == 5
    assert ens._species["l2e"]["ligand_stoich"] == {"iptg":4}
    assert ens._species["l2e"]["folded"] == True
    assert ens._species["l2e"]["observable"] == False

    assert ens._species["unfolded"]["dG0"] == 10
    assert ens._species["unfolded"]["ligand_stoich"] == {}
    assert ens._species["unfolded"]["folded"] == False
    assert ens._species["unfolded"]["observable"] == False

    # ens with no entries
    ens = _json_to_ensemble(ensemble_inputs["empty-ensemble.json"])
    assert len(ens._species) == 0
    assert ens._gas_constant == 0.008314

    # ensemble as top-level key
    ens = _json_to_ensemble(ensemble_inputs["top-level-ensemble.json"])
    assert len(ens._species) == 4
    assert ens._gas_constant == 0.008314

    # no gas constant defined
    ens = _json_to_ensemble(ensemble_inputs["no-gas-constant.json"])
    assert len(ens._species) == 4
    assert ens._gas_constant == 0.001987

    # no ensemble key at all -- die. 
    with pytest.raises(ValueError):
        ens = _json_to_ensemble(ensemble_inputs["no-ensemble.json"])

    with pytest.raises(ValueError):
        ens = _json_to_ensemble(ensemble_inputs["ensemble-with-bad-key.json"])

    current_dir = os.getcwd()
    os.chdir(tmpdir)

    # Load ensemble json where ens has "spreadsheet" key and non-default
    # gas constant. 

    shutil.copy(ensemble_inputs["s1s2_folded.xlsx"],".")
    ens = _json_to_ensemble(ensemble_inputs["spreadsheet-ensemble.json"])
    ens._gas_constant == 0.008314
    assert np.array_equal(ens.species,["s1","s2"])

    assert ens._species["s1"]["dG0"] == 0
    assert ens._species["s1"]["ligand_stoich"]["X"] == 1
    assert ens._species["s1"]["ligand_stoich"]["Y"] == 0
    assert ens._species["s1"]["folded"] == True
    assert ens._species["s1"]["observable"] == True
        
    assert ens._species["s2"]["dG0"] == 10
    assert ens._species["s2"]["ligand_stoich"]["X"] == 0
    assert ens._species["s2"]["ligand_stoich"]["Y"] == 1
    assert ens._species["s2"]["folded"] == True
    assert ens._species["s2"]["observable"] == False

    os.chdir(current_dir)


def test_read_ensemble(tmpdir,ensemble_inputs,variable_types):

    current_dir = os.getcwd()
    os.chdir(tmpdir)

    # Send in something stupid
    with pytest.raises(FileNotFoundError):
        read_ensemble("not_a_file")

    for v in variable_types["everything"]:
        print(v,type(v))
        with pytest.raises(FileNotFoundError):
            read_ensemble(v)

    # Send in json
    ens = read_ensemble(ensemble_inputs["top-level-ensemble.json"])
    assert ens._gas_constant == 0.008314
    assert len(ens.species) == 4

    # Send in excel
    ens = read_ensemble(ensemble_inputs["s1s2_folded.xlsx"])
    assert ens._gas_constant == 0.001987
    assert len(ens.species) == 2

    os.chdir(current_dir)