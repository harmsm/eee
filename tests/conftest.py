import pytest
import os
import glob

def _file_globber(*args):
    """
    Do a glob query constructed from *args relative to this directory. Return 
    a dictionary keying the basename of each file to its absolute path. 

    Example: 
    this/complicated/path/ has the files "one.csv", "two.csv", and "blah.txt".
    If args = ["this","complicated","path","*.csv"], the function will return
    {"one.csv":abspath("this/complicated/path/one.csv"),
     "two.csv":abspath("this/complicated/path/two.csv")}.
    """ 
    
    base_dir = os.path.dirname(os.path.realpath(__file__))
    search_string = os.path.join(base_dir,*args)

    file_dict = {}
    for g in glob.glob(search_string):
        key = os.path.basename(g)
        file_dict[key] = g 

    return file_dict

@pytest.fixture(scope="module")
def test_cifs():
    """
    Dictionary holding cif files for testing. 
    """

    return _file_globber("data","test_structures","*.cif")

@pytest.fixture(scope="module")
def test_pdbs():
    """
    Dictionary holding pdb files for testing. 
    """

    return _file_globber("data","test_structures","*.pdb")

@pytest.fixture(scope="module")
def ensembles():

    base_dir = os.path.dirname(os.path.realpath(__file__))
    search_string = os.path.join(base_dir,"data","ensembles","*")

    file_dict = {}
    for g in glob.glob(search_string):
        key = os.path.basename(g)
        rcsb_files = glob.glob(os.path.join(g,"*.*"))

        file_dict[key] = rcsb_files

    return file_dict

@pytest.fixture(scope="module")
def test_ddg():

    return _file_globber("data","test_ddg","*.csv")