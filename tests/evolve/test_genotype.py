
import pytest

from eee.evolve.genotype import Genotype
from eee.evolve.genotype import GenotypeContainer

import numpy as np
import pandas as pd

def test_Genotype(ens_test_data):

    ens = ens_test_data["ens"]
    ddg_dict = ens_test_data["ddg_dict"]

    # ---------------------------------
    # minimal input
    g = Genotype(ens=ens,
                 ddg_dict=ddg_dict)
    
    assert g._ens is ens
    assert g._ddg_dict is ddg_dict
    assert len(g.sites) == 0
    assert len(g.mutations) == 0
    assert len(g.mutations_accumulated) == 0
    assert len(g.mut_energy) == 2
    assert g.mut_energy["s1"] == 0
    assert g.mut_energy["s2"] == 0

    # ---------------------------------
    # sites, mutations, mut_energy, and mutations_accumulated input
    g = Genotype(ens=ens,
                 ddg_dict=ddg_dict,
                 sites=[1],
                 mutations=["M1A"],
                 mutations_accumulated=["M1A"],
                 mut_energy={"s1":1,"s2":-1})
    
    assert g._ens is ens
    assert g._ddg_dict is ddg_dict
    assert len(g.sites) == 1
    assert g.sites[0] == 1
    assert len(g.mutations) == 1
    assert g.mutations[0] == "M1A"
    assert len(g.mutations_accumulated) == 1
    assert g.mutations_accumulated[0] == "M1A"
    assert len(g.mut_energy) == 2
    assert g.mut_energy["s1"] == 1
    assert g.mut_energy["s2"] == -1


def test_Genotype_copy(ens_test_data):
    
    ens = ens_test_data["ens"]
    ddg_dict = ens_test_data["ddg_dict"]

    # ---------------------------------
    # Create a genotype object
    g = Genotype(ens=ens,
                 ddg_dict=ddg_dict,
                 sites=[1],
                 mutations=["M1A"],
                 mutations_accumulated=["M1A"],
                 mut_energy={"s1":1,"s2":-1})
    
    assert g._ens is ens
    assert g._ddg_dict is ddg_dict
    assert len(g.sites) == 1
    assert g.sites[0] == 1
    assert len(g.mutations) == 1
    assert g.mutations[0] == "M1A"
    assert len(g.mutations_accumulated) == 1
    assert g.mutations_accumulated[0] == "M1A"
    assert len(g.mut_energy) == 2
    assert g.mut_energy["s1"] == 1
    assert g.mut_energy["s2"] == -1

    # ---------------------------------
    # Now test copying of that genotype

    g2 = g.copy()

    # These should be the same object (copy by reference)
    assert g._ens is g2._ens
    assert g._ddg_dict is g2._ddg_dict

    # These should be different objects
    assert g._sites is not g2._sites
    assert g._mutations is not g2._mutations
    assert g._mutations_accumulated is not g2._mutations_accumulated
    assert g._mut_energy is not g2._mut_energy

    # But they should have copied contents correctly
    assert len(g2.sites) == 1
    assert g2.sites[0] == 1
    assert len(g2.mutations) == 1
    assert g2.mutations[0] == "M1A"
    assert len(g2.mutations_accumulated) == 1
    assert g2.mutations_accumulated[0] == "M1A"
    assert len(g2.mut_energy) == 2
    assert g2.mut_energy["s1"] == 1
    assert g2.mut_energy["s2"] == -1

def test_Genotype_mutate(ens_test_data):
    
    ens = ens_test_data["ens"]
    ddg_dict = ens_test_data["ddg_dict"]

    # ---------------------------------
    # Create a genotype object
    g = Genotype(ens=ens,
                 ddg_dict=ddg_dict,
                 sites=[1],
                 mutations=["M1A"],
                 mutations_accumulated=["M1A"],
                 mut_energy={"s1":1,"s2":-1})
    
    assert len(g.sites) == 1
    assert g.sites[0] == 1
    assert len(g.mutations) == 1
    assert g.mutations[0] == "M1A"
    assert len(g.mutations_accumulated) == 1
    assert g.mutations_accumulated[0] == "M1A"
    assert len(g.mut_energy) == 2
    assert g.mut_energy["s1"] == 1
    assert g.mut_energy["s2"] == -1

    # Copy and mutate new site, new mutation
    g2 = g.copy()
    g2.mutate(2,"P2R")
    assert len(g2.sites) == 2
    assert g2.sites[0] == 1
    assert g2.sites[1] == 2
    assert len(g2.mutations) == 2
    assert g2.mutations[0] == "M1A"
    assert g2.mutations[1] == "P2R"
    assert len(g2.mutations_accumulated) == 2
    assert g2.mutations_accumulated[0] == "M1A"
    assert g2.mutations_accumulated[1] == "P2R"
    assert len(g2.mut_energy) == 2
    assert g2.mut_energy["s1"] == 1
    assert g2.mut_energy["s2"] == 0 # 1 unit higher 

   # Copy and mutate same site, new mutation
    g3 = g.copy()
    g3.mutate(1,"M1V")
    assert len(g3.sites) == 1
    assert g3.sites[0] == 1
    assert len(g3.mutations) == 1
    assert g3.mutations[0] == "M1V"
    assert len(g3.mutations_accumulated) == 2
    assert g3.mutations_accumulated[0] == "M1A"
    assert g3.mutations_accumulated[1] == "A1V"
    assert len(g3.mut_energy) == 2
    assert g3.mut_energy["s1"] == -1
    assert g3.mut_energy["s2"] == 1

    # Copy and mutate same site, same mutation --> reversion
    g4 = g.copy()
    g4.mutate(1,"M1A")
    assert len(g4.sites) == 0
    assert len(g4.mutations) == 0
    assert len(g4.mutations_accumulated) == 2
    assert g4.mutations_accumulated[0] == "M1A"
    assert g4.mutations_accumulated[1] == "A1M"
    assert len(g4.mut_energy) == 2
    assert g4.mut_energy["s1"] == 0
    assert g4.mut_energy["s2"] == 0

    # Start at wildtype and mutate
    g = Genotype(ens=ens,
                 ddg_dict=ddg_dict)
    
    g.mutate(2,"P2R")
    assert len(g.sites) == 1
    assert g.sites[0] == 2
    assert len(g.mutations) == 1
    assert g.mutations[0] == "P2R"
    assert len(g.mutations_accumulated) == 1
    assert g.mutations_accumulated[0] == "P2R"
    assert len(g.mut_energy) == 2
    assert g.mut_energy["s1"] == 0
    assert g.mut_energy["s2"] == 1


def test_GenotypeContainer(ens_test_data):

    fc = ens_test_data["fc"]
    ddg_dict = ens_test_data["ddg_dict"]
    ddg_df = ens_test_data["ddg_df"]

    gc = GenotypeContainer(fc=fc,
                           ddg_df=ddg_df)

    # Make sure that ddg_df loaded correctly into ddg_dict
    for a in ddg_dict:
        for b in ddg_dict[a]:
            assert gc._ddg_dict[a][b] == ddg_dict[a][b]
            

    # Make sure possible sites and mutations at sites are correct
    assert np.array_equal(gc._possible_sites,[1,2])
    assert np.array_equal(gc._mutations_at_sites[1],["M1A","M1V"])
    assert np.array_equal(gc._mutations_at_sites[2],["P2R","P2Q"])

    # Make sure it created correct Genotype instance
    assert len(gc.genotypes) == 1
    assert issubclass(type(gc.genotypes[0]),Genotype)
    assert gc.genotypes[0]._ens is fc.ens
    assert gc.genotypes[0]._ddg_dict is gc._ddg_dict
    assert len(gc.genotypes[0].sites) == 0
    assert len(gc.genotypes[0].mutations) == 0
    assert len(gc.genotypes[0].mutations_accumulated) == 0
    assert len(gc.genotypes[0].mut_energy) == 2
    assert gc.genotypes[0].mut_energy["s1"] == 0
    assert gc.genotypes[0].mut_energy["s2"] == 0

    # Make sure trajectories and fitnesses are set up correctly. 
    assert len(gc.trajectories) == 1
    assert len(gc.trajectories[0]) == 1
    assert gc.trajectories[0][0] == 0
    assert len(gc.fitnesses) == 1
    assert gc.fitnesses[0] >= 0
    assert gc.fitnesses[0] <= 1


    # send in copy of ddg_df with mangled column names. Should throw a 
    # ValueError  
    bad_ddg_df = ddg_df.copy()
    new_columns = list(bad_ddg_df.columns)
    idx = new_columns.index("s1")
    new_columns[idx] = "not_a_species"
    bad_ddg_df.columns = new_columns

    with pytest.raises(ValueError):
        gc = GenotypeContainer(fc=fc,
                           ddg_df=bad_ddg_df)

def test_GenotypeContainer__create_ddg_dict(ens_test_data):

    fc = ens_test_data["fc"]
    ddg_dict = ens_test_data["ddg_dict"]
    ddg_df = ens_test_data["ddg_df"]

    gc = GenotypeContainer(fc=fc,ddg_df=ddg_df)
    
    # Make sure that ddg_df loaded correctly into ddg_dict
    for a in ddg_dict:
        for b in ddg_dict[a]:
            assert gc._ddg_dict[a][b] == ddg_dict[a][b]

    # send in copy of ddg_df with mangled column names. Should throw a 
    # ValueError  
    bad_ddg_df = ddg_df.copy()
    new_columns = list(bad_ddg_df.columns)
    idx = new_columns.index("s1")
    new_columns[idx] = "not_a_species"
    bad_ddg_df.columns = new_columns

    with pytest.raises(ValueError):
        gc = GenotypeContainer(fc=fc,
                           ddg_df=bad_ddg_df)

def test_GenotypeContainer_mutate(ens_test_data):

    fc = ens_test_data["fc"]
    ddg_df = ens_test_data["ddg_df"]

    gc = GenotypeContainer(fc=fc,
                           ddg_df=ddg_df)
    
    # Mutate residue not there. 
    with pytest.raises(IndexError):
        gc.mutate(1)

    # Mutate residue not there. 
    with pytest.raises(IndexError):
        gc.mutate(-1)

    # Chain of mutations...
    allowable_muts =set(ddg_df["mut"])
    for i in range(4):

        gc.mutate(i)

        # Make sure trajectories accumulating as we expect
        assert len(gc.trajectories) == i + 2
        assert len(gc.trajectories[-1]) == i + 2
        assert np.array_equal(gc.trajectories[-1],
                              np.arange(i+2,dtype=int))
        
        # make sure fitness values are in proper bounds
        assert len(gc.fitnesses) == i + 2
        for f in gc.fitnesses:
            assert f >= 0 and f <= 1

        # Make sure we are accumulating genotypes objects that are not the same
        assert len(gc.genotypes) == i + 2
        for j in range(len(gc.genotypes)):
            assert issubclass(type(gc.genotypes[j]),Genotype)
            assert set(gc.genotypes[j].mutations).issubset(allowable_muts)

            for k in range(j+1,len(gc.genotypes)):
                assert gc.genotypes[j] is not gc.genotypes[k]



def test_GenotypeContainer_genotypes(ens_test_data):
    
    fc = ens_test_data["fc"]
    ddg_df = ens_test_data["ddg_df"]

    gc = GenotypeContainer(fc=fc,
                           ddg_df=ddg_df)
    gc.mutate(0)

    assert len(gc.genotypes) == 2
    assert issubclass(type(gc.genotypes[0]),Genotype)
    assert issubclass(type(gc.genotypes[1]),Genotype)
    assert gc.genotypes[0] is not gc.genotypes[1]

def test_GenotypeContainer_trajectories(ens_test_data):

    fc = ens_test_data["fc"]
    ddg_df = ens_test_data["ddg_df"]

    gc = GenotypeContainer(fc=fc,
                           ddg_df=ddg_df)
    gc.mutate(0)
    assert len(gc.trajectories) == 2
    assert len(gc.trajectories[0]) == 1
    assert gc.trajectories[0][0] == 0

    assert len(gc.trajectories[1]) == 2
    assert gc.trajectories[1][0] == 0
    assert gc.trajectories[1][1] == 1

def test_GenotypeContainer_fitnesses(ens_test_data):
    
    fc = ens_test_data["fc"]
    ddg_df = ens_test_data["ddg_df"]

    gc = GenotypeContainer(fc=fc,
                           ddg_df=ddg_df)
    gc.mutate(0)
    assert len(gc.fitnesses) == 2
    

def test_GenotypeContainer_df(ens_test_data):

    # not a great test of all features of dataframe. Makes sure table has right
    # columns and that mutational parent tracking works. 

    fc = ens_test_data["fc"]
    ddg_df = ens_test_data["ddg_df"]

    gc = GenotypeContainer(fc=fc,
                           ddg_df=ddg_df)
    
    assert issubclass(type(gc.df),pd.DataFrame)
    assert np.array_equal(gc.df.columns,
                          ["genotype",
                           "mutations","num_mutations",
                           "accum_mut","num_accum_mut",
                           "parent","trajectory","fitness"])
    
    assert len(gc.df) == len(gc.genotypes)

    gc.mutate(0)
    gc.mutate(0)
    gc.mutate(1)
    gc.mutate(3)

    # Drop first row from comparison because null causes problems
    assert np.array_equal(list(gc.df["parent"])[1:],[0,0,1,3])
    assert gc.df.loc[3,"num_accum_mut"] == 2


