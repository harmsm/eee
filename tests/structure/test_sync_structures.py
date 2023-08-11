
import pytest

from eee.structure.sync_structures import _create_unique_filenames
from eee.structure.sync_structures import sync_structures

import numpy as np

import os


def test__create_unique_filenames():

    files = ["1stn.pdb",
             "../test/1stn.pdb",
             "../test/this/1stn.pdb"]
    
    mapper = _create_unique_filenames(files)
    assert mapper["1stn.pdb"] == "1stn.pdb"
    assert mapper["../test/1stn.pdb"] == "test__1stn.pdb"
    assert mapper["../test/this/1stn.pdb"] == "this__1stn.pdb"

    files = ["../lab/1stn.pdb","../rocket/1stn.cif"]
    mapper = _create_unique_filenames(files)
    assert mapper["../lab/1stn.pdb"] == "1stn.pdb"
    assert mapper["../rocket/1stn.cif"] == "1stn.cif"

    files = ["../test/1stn.pdb",
             "../test/1stn.pdb",
             "../test/this/1stn.pdb"]
    with pytest.raises(ValueError):
        _create_unique_filenames(files)

def test_sync_structures(structure_ensembles,tmpdir):

    current_dir = os.getcwd()
    os.chdir(tmpdir)

    dfs = sync_structures(structure_ensembles["missing_residues"],
                          "missing_residues")

    os.chdir(current_dir)

    pass