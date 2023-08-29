
from eee.io.write_pdb import write_pdb
from eee.io.read_structure import _read_structure_pdb

import numpy as np

import pytest
import os

def test_write_pdb(test_pdbs,tmpdir):

    current_dir = os.getcwd()
    os.chdir(tmpdir)

    df = _read_structure_pdb(test_pdbs["1stn.pdb"])
    write_pdb(df,"test.pdb")

    # Load pdb and write again. This should now be identical
    new_df = _read_structure_pdb("test.pdb")
    write_pdb(new_df,"test2.pdb")
    new_df2 = _read_structure_pdb("test2.pdb")

    assert np.sum(np.array(new_df2 != new_df)) == 0

    # Check overrwrite flag
    with pytest.raises(FileExistsError):
        write_pdb(new_df,"test2.pdb")

    with pytest.raises(FileExistsError):
        write_pdb(new_df,"test2.pdb",overwrite=False)

    write_pdb(new_df,"test2.pdb",overwrite=True)



    os.chdir(current_dir)
    