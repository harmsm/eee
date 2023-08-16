import pytest

from eee.ensemble import Ensemble

import numpy as np
import pandas as pd


def test_Ensemble():

    ens = Ensemble()
    assert issubclass(type(ens),Ensemble)
    
    ens = Ensemble(R=1)
    assert ens._R == 1


def test_Ensemble_add_species(variable_types):
    
    ens = Ensemble()
    ens.add_species(name="test",
                    observable=False,
                    folded=True,
                    dG0=-1,
                    mu_stoich=None)
    
    assert ens._species["test"]["observable"] == False
    assert ens._species["test"]["folded"] == True
    assert ens._species["test"]["dG0"] == -1
    assert ens._species["test"]["mu_stoich"] == {}
    assert np.array_equal(ens._mu_list,[])

    # Check for something already in ensemble
    with pytest.raises(ValueError):
        ens.add_species(name="test",
                        observable=False,
                        folded=True,
                        dG0=-1,
                        mu_stoich=None)
        
    ens.add_species(name="another",
                    observable=True,
                    folded=False,
                    mu_stoich={"X":1})
    
    assert ens._species["another"]["observable"] == True
    assert ens._species["another"]["folded"] == False
    assert ens._species["another"]["dG0"] == 0
    assert ens._species["another"]["mu_stoich"]["X"] == 1
    assert np.array_equal(ens._mu_list,["X"])

    # observable argument type checking
    print("--- observable ---")
    for v in variable_types["bools"]:
        print(v,type(v))
        ens = Ensemble()
        ens.add_species(name="test",observable=v)
        assert ens._species["test"]["observable"] == bool(v)

    for v in variable_types["not_bools"]:
        print(v,type(v))
        with pytest.raises(ValueError):
            ens.add_species(name="test",observable=v)

    # folded argument type checking
    print("--- folded ---")
    for v in variable_types["bools"]:
        print(v,type(v))
        ens = Ensemble()
        ens.add_species(name="test",folded=v)
        assert ens._species["test"]["folded"] == bool(v)

    for v in variable_types["not_bools"]:
        print(v,type(v))
        with pytest.raises(ValueError):
            ens.add_species(name="test",folded=v)

    # dG0 argument type checking
    print("--- dG0 ---")
    for v in variable_types["floats_or_coercable"]:
        print(v,type(v))
        ens = Ensemble()
        ens.add_species(name="test",dG0=v)
        assert ens._species["test"]["dG0"] == float(v)

    for v in variable_types["not_floats_or_coercable"]:
        print(v,type(v))
        ens = Ensemble()
        with pytest.raises(ValueError):
            ens.add_species(name="test",dG0=v)

    # mu_dict argument type checking
    print("--- mu_dict ---")
    for v in variable_types["dict"]:
        print(v,type(v))
        ens = Ensemble()
        ens.add_species(name="test",mu_stoich=v)

    for v in variable_types["not_dict"]:
        print(v,type(v))

        # None is allowed
        if v is None:
            continue

        ens = Ensemble()
        with pytest.raises(ValueError):
            ens.add_species(name="test",mu_stoich=v)


def test_Ensemble__build_z_matrix():

    # Single species, not observable, dG = 0, not coupled to mu
    ens = Ensemble()
    ens.add_species(name="test1",
                    observable=False,
                    folded=True,
                    dG0=0,
                    mu_stoich=None)

    ens._build_z_matrix(mu_dict={})
    assert np.array_equal(ens._z_matrix.shape,(1,1))
    assert np.array_equal(ens._z_matrix,[[0.0]])
    assert np.array_equal(ens._obs_mask,[False])
    assert np.array_equal(ens._not_obs_mask,[True])
    assert np.array_equal(ens._folded_mask,[True])
    assert np.array_equal(ens._unfolded_mask,[False])

    # Single species, not observable, dG = 0, coupled to mu
    ens = Ensemble()
    ens.add_species(name="test1",
                    observable=False,
                    folded=True,
                    dG0=0,
                    mu_stoich={"X":1})

    ens._build_z_matrix(mu_dict={})
    assert np.array_equal(ens._z_matrix.shape,(1,1))
    assert np.array_equal(ens._z_matrix,[[0.0]])
    assert np.array_equal(ens._obs_mask,[False])
    assert np.array_equal(ens._not_obs_mask,[True])
    assert np.array_equal(ens._folded_mask,[True])
    assert np.array_equal(ens._unfolded_mask,[False])

    # Single species, not observable, dG = 0, coupled to mu. Now add mu
    ens = Ensemble()
    ens.add_species(name="test1",
                    observable=False,
                    folded=True,
                    dG0=0,
                    mu_stoich={"X":1})

    ens._build_z_matrix(mu_dict={"X":np.array([1.0])})
    assert np.array_equal(ens._z_matrix.shape,(1,1))
    assert np.array_equal(ens._z_matrix,[[-1]])
    assert np.array_equal(ens._obs_mask,[False])
    assert np.array_equal(ens._not_obs_mask,[True])
    assert np.array_equal(ens._folded_mask,[True])
    assert np.array_equal(ens._unfolded_mask,[False])

    # Two species, one observable, dG = 0, One coupled to mu, with single mu
    ens = Ensemble()
    ens.add_species(name="test1",
                    observable=False,
                    folded=True,
                    dG0=0,
                    mu_stoich={"X":1})
    ens.add_species(name="test2",
                    observable=True,
                    folded=False,
                    dG0=0)

    ens._build_z_matrix(mu_dict={"X":np.array([1.0])})
    assert np.array_equal(ens._z_matrix.shape,(2,1))
    assert np.array_equal(ens._z_matrix,[[-1],[0]])
    assert np.array_equal(ens._obs_mask,[False,True])
    assert np.array_equal(ens._not_obs_mask,[True,False])
    assert np.array_equal(ens._folded_mask,[True,False])
    assert np.array_equal(ens._unfolded_mask,[False,True])

    # Two species, one observable, dG = 0, One coupled to mu, with three mu
    ens = Ensemble()
    ens.add_species(name="test1",
                    observable=False,
                    folded=True,
                    dG0=0,
                    mu_stoich={"X":1})
    ens.add_species(name="test2",
                    observable=True,
                    folded=False,
                    dG0=0)

    ens._build_z_matrix(mu_dict={"X":np.array([0,0.5,1.0])})
    assert np.array_equal(ens._z_matrix.shape,(2,3))
    assert np.array_equal(ens._z_matrix,[[0,-0.5,-1],[0,0,0]])
    assert np.array_equal(ens._obs_mask,[False,True])
    assert np.array_equal(ens._not_obs_mask,[True,False])
    assert np.array_equal(ens._folded_mask,[True,False])
    assert np.array_equal(ens._unfolded_mask,[False,True])
    
    # Two species, one observable, dG = 0, 1, One coupled to mu, with three mu
    ens = Ensemble()
    ens.add_species(name="test1",
                    observable=False,
                    folded=True,
                    dG0=0,
                    mu_stoich={"X":1})
    ens.add_species(name="test2",
                    observable=True,
                    folded=False,
                    dG0=1)

    ens._build_z_matrix(mu_dict={"X":np.array([0,0.5,1.0])})
    assert np.array_equal(ens._z_matrix.shape,(2,3))
    assert np.array_equal(ens._z_matrix,[[0,-0.5,-1],[1,1,1]])
    assert np.array_equal(ens._obs_mask,[False,True])
    assert np.array_equal(ens._not_obs_mask,[True,False])
    assert np.array_equal(ens._folded_mask,[True,False])
    assert np.array_equal(ens._unfolded_mask,[False,True])

    # Two species, one observable, dG = 0, 1, Both coupled to different mu, 
    # with three mu
    ens = Ensemble()
    ens.add_species(name="test1",
                    observable=False,
                    folded=True,
                    dG0=0,
                    mu_stoich={"X":1})
    ens.add_species(name="test2",
                    observable=True,
                    folded=False,
                    dG0=1,
                    mu_stoich={"Y":2})

    ens._build_z_matrix(mu_dict={"X":np.array([0,0.5,1.0]),
                                 "Y":np.array([1,0.5,0.0])})
    assert np.array_equal(ens._z_matrix.shape,(2,3))
    assert np.array_equal(ens._z_matrix,[[0,-0.5,-1],[-2 + 1,-1 + 1,0 + 1]])
    assert np.array_equal(ens._obs_mask,[False,True])
    assert np.array_equal(ens._not_obs_mask,[True,False])
    assert np.array_equal(ens._folded_mask,[True,False])
    assert np.array_equal(ens._unfolded_mask,[False,True])


    # Three species, dG = 0, 1, 3 Two coupled to different mu,  with three mu
    ens = Ensemble()
    ens.add_species(name="test1",
                    observable=False,
                    folded=True,
                    dG0=0,
                    mu_stoich={"X":1})
    ens.add_species(name="test2",
                    observable=True,
                    folded=False,
                    dG0=1,
                    mu_stoich={"Y":2})
    ens.add_species(name="test3",
                    observable=True,
                    folded=True,
                    dG0=3)

    ens._build_z_matrix(mu_dict={"X":np.array([0,0.5,1.0]),
                                 "Y":np.array([1,0.5,0.0])})
    assert np.array_equal(ens._z_matrix.shape,(3,3))
    assert np.array_equal(ens._z_matrix,[[0,-0.5,-1],
                                         [-2 + 1,-1 + 1,0 + 1],
                                         [3,3,3]])
    assert np.array_equal(ens._obs_mask,[False,True,True])
    assert np.array_equal(ens._not_obs_mask,[True,False,False])
    assert np.array_equal(ens._folded_mask,[True,False,True])
    assert np.array_equal(ens._unfolded_mask,[False,True,False])

def test_Ensemble__get_weights():

    # single species, R = 1, T = 1, no mutations
    ens = Ensemble(R=1)
    ens.add_species(name="test1",
                    observable=False,
                    folded=True,
                    dG0=0,
                    mu_stoich=None)
    
    mut_energy = np.array([0.0])
    T = np.ones(1,dtype=float)    

    ens._build_z_matrix(mu_dict={})
    weights = ens._get_weights(mut_energy=mut_energy,T=T)
    assert np.array_equal(weights.shape,[1,1])
    assert np.array_equal(weights,[np.exp([ens._max_allowed])])

    # single species, R = 1, T = 500, no mutations. This tests the shift to 
    # max_allowed. 
    ens = Ensemble(R=1)
    ens.add_species(name="test1",
                    observable=False,
                    folded=True,
                    dG0=0,
                    mu_stoich=None)
    
    mut_energy = np.array([0.0])
    T = 500*np.ones(1,dtype=float)    
    ens._build_z_matrix(mu_dict={})
    weights = ens._get_weights(mut_energy=mut_energy,T=T)
    assert np.array_equal(weights.shape,[1,1])
    assert np.array_equal(weights,[np.exp([ens._max_allowed])])
    
    # Two species. 
    ens = Ensemble(R=1)
    ens.add_species(name="test1",
                    observable=False,
                    folded=True,
                    dG0=0,
                    mu_stoich=None)
    ens.add_species(name="test2",
                    observable=False,
                    folded=False,
                    dG0=0,
                    mu_stoich=None)
    
    mut_energy = np.array([0.0,0.0])
    T = np.ones(1,dtype=float)    
    ens._build_z_matrix(mu_dict={})
    weights = ens._get_weights(mut_energy=mut_energy,T=T)
    assert np.array_equal(weights.shape,[2,1])
    assert np.array_equal(weights,[[np.exp(ens._max_allowed)],
                                   [np.exp(ens._max_allowed)]])


    # Two species. Mutate one. 
    ens = Ensemble(R=1)
    ens.add_species(name="test1",
                    observable=False,
                    folded=True,
                    dG0=0,
                    mu_stoich=None)
    ens.add_species(name="test2",
                    observable=False,
                    folded=False,
                    dG0=0,
                    mu_stoich=None)
    
    mut_energy = np.array([0.0,1.0])
    T = np.ones(1,dtype=float)    
    ens._build_z_matrix(mu_dict={})
    weights = ens._get_weights(mut_energy=mut_energy,T=T)
    assert np.array_equal(weights.shape,[2,1])
    norm_weights = weights/np.sum(weights,axis=0)
    Z = np.exp(0) + np.exp(-1)
    assert np.isclose(norm_weights[0,0],np.exp(0)/Z)
    assert np.isclose(norm_weights[1,0],np.exp(-1)/Z)

    # Make sure temperature works as expected
    T = 50*np.ones(1,dtype=float)                                        
    weights = ens._get_weights(mut_energy=mut_energy,T=T)
    assert np.array_equal(weights.shape,[2,1])
    norm_weights = weights/np.sum(weights,axis=0)
    Z = np.exp(0/50) + np.exp(-1/50)
    assert np.isclose(norm_weights[0,0],np.exp(0/50)/Z)
    assert np.isclose(norm_weights[1,0],np.exp(-1/50)/Z)
                                        

    # Two species. Mutate one. Alter R to test gas constant
    ens = Ensemble(R=50)
    ens.add_species(name="test1",
                    observable=False,
                    folded=True,
                    dG0=0,
                    mu_stoich=None)
    ens.add_species(name="test2",
                    observable=False,
                    folded=False,
                    dG0=0,
                    mu_stoich=None)
    
    mut_energy = np.array([0.0,1.0])
    T = np.ones(1,dtype=float)*50    
    ens._build_z_matrix(mu_dict={})
                                  
    weights = ens._get_weights(mut_energy=mut_energy,T=T)
    assert np.array_equal(weights.shape,[2,1])
    norm_weights = weights/np.sum(weights,axis=0)
    Z = np.exp(0/2500) + np.exp(-1/2500)
    assert np.isclose(norm_weights[0,0],np.exp(0/2500)/Z)
    assert np.isclose(norm_weights[1,0],np.exp(-1/2500)/Z)


    # Two species. dG0
    ens = Ensemble(R=1)
    ens.add_species(name="test1",
                    observable=False,
                    folded=True,
                    dG0=-1,
                    mu_stoich=None)
    ens.add_species(name="test2",
                    observable=False,
                    folded=False,
                    dG0=0,
                    mu_stoich=None)
    
    mut_energy = np.array([0.0,1.0])
    T = np.ones(1,dtype=float)    
    ens._build_z_matrix(mu_dict={})
                        
    weights = ens._get_weights(mut_energy=mut_energy,T=T)
    assert np.array_equal(weights.shape,[2,1])
    norm_weights = weights/np.sum(weights,axis=0)
    Z = np.exp(1) + np.exp(-1)
    assert np.isclose(norm_weights[0,0],np.exp(1)/Z)
    assert np.isclose(norm_weights[1,0],np.exp(-1)/Z)

    # Two species. dG0. Add mu_dict perturbation
    ens = Ensemble(R=1)
    ens.add_species(name="test1",
                    observable=False,
                    folded=True,
                    dG0=-1,
                    mu_stoich={"X":1})
    ens.add_species(name="test2",
                    observable=False,
                    folded=False,
                    dG0=0,
                    mu_stoich=None)
    
    mut_energy = np.array([0.0,1.0])
    T = np.ones(1,dtype=float)    
    ens._build_z_matrix(mu_dict={"X":np.array([0,0.5,1])})
                        
    weights = ens._get_weights(mut_energy=mut_energy,T=T)
    assert np.array_equal(weights.shape,[2,3])
    norm_weights = weights/np.sum(weights,axis=0)
    Z = np.exp(1) + np.exp(-1)
    assert np.isclose(norm_weights[0,0],np.exp(1)/Z)
    assert np.isclose(norm_weights[1,0],np.exp(-1)/Z)

    Z = np.exp(1 + 0.5) + np.exp(-1)
    assert np.isclose(norm_weights[0,1],np.exp(1+0.5)/Z)
    assert np.isclose(norm_weights[1,1],np.exp(-1)/Z)

    Z = np.exp(1 + 1.0) + np.exp(-1)
    assert np.isclose(norm_weights[0,2],np.exp(1+1.0)/Z)
    assert np.isclose(norm_weights[1,2],np.exp(-1)/Z)

def test_load_mu_dict(variable_types):

    # Two species. dG0. Add mu_dict perturbation
    ens = Ensemble()
    ens.add_species(name="test1",
                    observable=False,
                    folded=True,
                    dG0=0,
                    mu_stoich={"X":1})
    ens.add_species(name="test2",
                    observable=True,
                    folded=False,
                    dG0=1,
                    mu_stoich={"Y":2})
    
    for v in variable_types["not_dict"]:
        print(v,type(v))
        with pytest.raises(ValueError):
            ens.load_mu_dict(v)

    assert not hasattr(ens,"_z_matrix")

    # Just check one load -- wraps _create_z_matrix which we already test 
    # extensively. 
    ens.load_mu_dict(mu_dict={"X":np.array([0,0.5,1.0]),
                              "Y":np.array([1,0.5,0.0])})
    assert np.array_equal(ens._z_matrix.shape,(2,3))
    assert np.array_equal(ens._z_matrix,[[0,-0.5,-1],[-2 + 1,-1 + 1,0 + 1]])
    assert np.array_equal(ens._obs_mask,[False,True])
    assert np.array_equal(ens._not_obs_mask,[True,False])
    assert np.array_equal(ens._folded_mask,[True,False])
    assert np.array_equal(ens._unfolded_mask,[False,True])


def test_mut_dict_to_array():

    # Two species. 
    ens = Ensemble(R=1)
    ens.add_species(name="test1",
                    observable=False,
                    dG0=0,
                    mu_stoich=None)
    ens.add_species(name="test2",
                    observable=False,
                    dG0=0,
                    mu_stoich=None)
    
    out_array = ens.mut_dict_to_array({"test1":1.0,"test2":2.0})
    assert np.array_equal(out_array,[1,2])

    out_array = ens.mut_dict_to_array({"test2":1.0,"test1":2.0})
    assert np.array_equal(out_array,[2,1])

    # Hack that should invert outputs. Never really happen in real life.
    ens._species_list = ["test2","test1"]
    out_array = ens.mut_dict_to_array({"test2":1.0,"test1":2.0})
    assert np.array_equal(out_array,[1,2])

def test_get_fx_obs_fast():

    ens = Ensemble(R=1)
    ens.add_species(name="test1",
                    observable=False,
                    folded=True,
                    dG0=1,
                    mu_stoich={"X":1})
    ens.add_species(name="test2",
                    observable=True,
                    folded=False,
                    dG0=0)
    ens.add_species(name="test3",
                    observable=True,
                    folded=True,
                    dG0=2)

    ens.load_mu_dict(mu_dict={"X":np.array([0,1.0])})
    T = np.ones(1,dtype=float)    

    value, fx_folded = ens.get_fx_obs_fast(mut_energy_array=np.array([0,0,0]),T=T)
    t1 = np.exp(-1)
    t2 = np.exp(-0)
    t3 = np.exp(-2)
    Z = t1 + t2 + t3
    predicted = [(t2 + t3)/Z]
    t1 = np.exp(0)
    t2 = np.exp(-0)
    t3 = np.exp(-2)
    Z = t1 + t2 + t3
    predicted.append((t2 + t3)/Z)

    assert np.array_equal(np.round(value,2),
                          np.round(predicted,2))

    t1 = np.exp(-1) 
    t2 = np.exp(-0)
    t3 = np.exp(-2)
    Z = t1 + t2 + t3
    predicted = [(t1 + t3)/Z]
    t1 = np.exp(0)
    t2 = np.exp(-0)
    t3 = np.exp(-2)
    Z = t1 + t2 + t3
    predicted.append((t1 + t3)/Z)

    assert np.array_equal(np.round(fx_folded,2),
                          np.round(predicted,2))

    value, fx_folded = ens.get_fx_obs_fast(mut_energy_array=np.array([0,-1,0]),T=T)
    t1 = np.exp(-1)
    t2 = np.exp(1)
    t3 = np.exp(-2)
    Z = t1 + t2 + t3
    predicted = [(t2 + t3)/Z]
    t1 = np.exp(0)
    t2 = np.exp(1)
    t3 = np.exp(-2)
    Z = t1 + t2 + t3
    predicted.append((t2 + t3)/Z)

    assert np.array_equal(np.round(value,2),
                          np.round(predicted,2))
    
    t1 = np.exp(-1)
    t2 = np.exp(1)
    t3 = np.exp(-2)
    Z = t1 + t2 + t3
    predicted = [(t1 + t3)/Z]
    t1 = np.exp(0)
    t2 = np.exp(1)
    t3 = np.exp(-2)
    Z = t1 + t2 + t3
    predicted.append((t1 + t3)/Z)

    assert np.array_equal(np.round(fx_folded,2),
                          np.round(predicted,2))


def test_get_dG_obs_fast():

    
    ens = Ensemble(R=1)
    ens.add_species(name="test1",
                    observable=False,
                    folded=True,
                    dG0=1,
                    mu_stoich={"X":1})
    ens.add_species(name="test2",
                    observable=True,
                    folded=False,
                    dG0=0)
    ens.add_species(name="test3",
                    observable=True,
                    folded=True,
                    dG0=2)

    ens.load_mu_dict(mu_dict={"X":np.array([0,1.0])})
    T = np.ones(1,dtype=float)    

    value, fx_folded = ens.get_dG_obs_fast(mut_energy_array=np.array([0,0,0]),T=T)
    t1 = np.exp(-1)
    t2 = np.exp(-0)
    t3 = np.exp(-2)
    predicted = [-np.log((t2 + t3)/t1)]

    t1 = np.exp(0)
    t2 = np.exp(-0)
    t3 = np.exp(-2)
    predicted.append(-np.log((t2 + t3)/t1))

    assert np.array_equal(np.round(value,2),
                          np.round(predicted,2))
    
    t1 = np.exp(-1)
    t2 = np.exp(-0)
    t3 = np.exp(-2)
    predicted = [(t1 + t3)/(t1+t2+t3)]

    t1 = np.exp(0)
    t2 = np.exp(-0)
    t3 = np.exp(-2)
    predicted.append((t1 + t3)/(t1+t2+t3))

    assert np.array_equal(np.round(fx_folded,2),
                          np.round(predicted,2))
    

    value, fx_folded = ens.get_dG_obs_fast(mut_energy_array=np.array([0,-1,0]),T=T)
    t1 = np.exp(-1)
    t2 = np.exp(1)
    t3 = np.exp(-2)
    predicted = [-np.log((t2 + t3)/t1)]

    t1 = np.exp(0)
    t2 = np.exp(1)
    t3 = np.exp(-2)
    predicted.append(-np.log((t2 + t3)/t1))

    assert np.array_equal(np.round(value,2),
                          np.round(predicted,2))

    t1 = np.exp(-1)
    t2 = np.exp(1)
    t3 = np.exp(-2)
    predicted = [(t1 + t3)/(t1+t2+t3)]

    t1 = np.exp(0)
    t2 = np.exp(1)
    t3 = np.exp(-2)
    predicted.append((t1 + t3)/(t1+t2+t3))

    assert np.array_equal(np.round(fx_folded,2),
                          np.round(predicted,2))

def test_Ensemble_get_species_dG(variable_types):

    ens = Ensemble()
    ens.add_species(name="test",
                    observable=False,
                    folded=True,
                    dG0=-1,
                    mu_stoich=None)

    with pytest.raises(ValueError):
        ens.get_species_dG("not_a_species")
    
    assert ens.get_species_dG("test") == -1
    assert ens.get_species_dG("test",mut_energy=10) == 9
    assert ens.get_species_dG("test",mut_energy=10,mu_dict={"X":10}) == 9

    ens.add_species(name="another",
                    observable=True,
                    folded=False,
                    mu_stoich={"X":1})
    
    assert ens.get_species_dG("another") == 0
    assert ens.get_species_dG("another",mut_energy=10) == 10
    assert ens.get_species_dG("another",mut_energy=10,mu_dict={"X":10}) == 0

    # Pass in an array of mu X
    value = ens.get_species_dG("another",mut_energy=10,mu_dict={"X":np.arange(10)})
    assert np.array_equal(value,-np.arange(10) + 10)

    # Stoichiometry of 2
    ens = Ensemble()
    ens.add_species(name="stoich2",
                    observable=False,
                    folded=True,
                    dG0=-5,
                    mu_stoich={"X":2})
    value = ens.get_species_dG("stoich2",mut_energy=10,mu_dict={"X":np.arange(10)})
    assert np.array_equal(value,-np.arange(10)*2 + 5)

    # Two chemical potentials same species
    ens = Ensemble()
    ens.add_species(name="two_mu",
                    observable=False,
                    folded=True,
                    dG0=-5,
                    mu_stoich={"X":2,"Y":1})
    value = ens.get_species_dG("two_mu",
                               mut_energy=0,
                               mu_dict={"X":np.arange(10),
                                        "Y":np.arange(10)})
    assert np.array_equal(value,-np.arange(10)*2+-np.arange(10)-5)

    # mut_energy argument type checking
    print("--- mut_energy ---")
    for v in variable_types["floats_or_coercable"]:
        print(v,type(v))
        ens = Ensemble()
        ens.add_species(name="test")
        assert ens.get_species_dG(name="test",mut_energy=v) == float(v)

    for v in variable_types["not_floats_or_coercable"]:
        print(v,type(v))
        ens = Ensemble()
        ens.add_species(name="test")
        with pytest.raises(ValueError):
            ens.get_species_dG(name="test",mut_energy=v)

    # mu_dict argument type checking
    print("--- mu_dict ---")
    for v in variable_types["dict"]:
        print(v,type(v))
        ens = Ensemble()
        ens.add_species(name="test1")
        ens.add_species(name="test2",observable=True)
        ens.get_obs(mu_dict=v)

    for v in variable_types["not_dict"]:
        print(v,type(v))

        # none okay
        if v is None:
            continue

        ens = Ensemble()
        ens.add_species(name="test1")
        ens.add_species(name="test2",observable=True)
        with pytest.raises(ValueError):
            ens.get_obs(mu_dict=v)

    for v in variable_types["float_value_or_iter"]:
        print(v,type(v))
        
        if hasattr(v,"__iter__") and len(v) == 0:
            continue

        if issubclass(type(v),pd.DataFrame):
            continue

        mu_dict = {"X":v}
        ens = Ensemble()
        ens.add_species(name="test1")
        ens.add_species(name="test2",observable=True)
        ens.get_obs(mu_dict=mu_dict)

    not_allowed = variable_types["not_float_value_or_iter"][:]
    not_allowed.append([])
    not_allowed.append(pd.DataFrame({"X":[1,2,3]}))

    for v in not_allowed:
        print(v,type(v))
    
        mu_dict = {"X":v}
        ens = Ensemble()
        ens.add_species(name="test1")
        ens.add_species(name="test2",observable=True)
        with pytest.raises(ValueError):
            ens.get_obs(mu_dict=mu_dict)


def test_Ensemble_get_obs(variable_types):

    # ------------------------------------------------------------------------
    # Not enough species
    ens = Ensemble()
    ens.add_species(name="test",
                    folded=True,
                    observable=False)
    with pytest.raises(ValueError):
        ens.get_obs()

    # ------------------------------------------------------------------------
    # No observable species
    ens = Ensemble()
    ens.add_species(name="test1",
                    folded=True,
                    observable=False)
    ens.add_species(name="test2",
                    folded=False,
                    observable=False)
    with pytest.raises(ValueError):
        ens.get_obs()

    # ------------------------------------------------------------------------
    # Only observable species
    ens = Ensemble()
    ens.add_species(name="test1",
                    observable=True,
                    folded=True)
    ens.add_species(name="test2",
                    observable=True,
                    folded=False)
    with pytest.raises(ValueError):
        ens.get_obs()

    # ------------------------------------------------------------------------
    # One observable, one not
    ens = Ensemble()
    ens.add_species(name="test1",
                    observable=True,
                    folded=True)
    ens.add_species(name="test2",
                    observable=False,
                    folded=False)
    
    df = ens.get_obs()
    assert df.loc[0,"dG_obs"] == 0
    assert df.loc[0,"fx_obs"] == 0.5
    assert df.loc[0,"test1"] == 0.5
    assert df.loc[0,"test2"] == 0.5
    assert df.loc[0,"fx_folded"] == 0.5

    # ------------------------------------------------------------------------
    # One observable, two not. (Use R = 1 and T = 1 to simplify math)
    ens = Ensemble(R=1)
    ens.add_species(name="test1",
                    observable=True,
                    folded=False)
    ens.add_species(name="test2",
                    observable=False,
                    folded=True)
    ens.add_species(name="test3",   
                    observable=False,
                    folded=True)
    
    df = ens.get_obs(T=1)
    assert df.loc[0,"dG_obs"] == -np.log(1/2)
    assert df.loc[0,"fx_obs"] == 1/3
    assert df.loc[0,"test1"] == 1/3
    assert df.loc[0,"test2"] == 1/3
    assert df.loc[0,"fx_folded"] == 2/3

    # ------------------------------------------------------------------------
    # One observable, two not. (Use R = 1 and T = 1 to simplify math). mu_dict
    # interesting.
    ens = Ensemble(R=1)
    ens.add_species(name="test1",
                    observable=True,
                    folded=False,
                    mu_stoich={"X":1})
    ens.add_species(name="test2",
                    observable=False,
                    folded=True)
    ens.add_species(name="test3",   
                    observable=False,
                    folded=True)
    df = ens.get_obs(T=1,
                     mu_dict={"X":[0,1]})
    
    test1 = np.exp(0)
    test2 = np.exp(0)
    test3 = np.exp(0)
    test_all = test1 + test2 + test3
    numerator = test1
    denominator = test2 + test3

    dG = -np.log(numerator/denominator)
    fx = numerator/(numerator + denominator)

    assert np.isclose(df.loc[0,"dG_obs"],dG)
    assert np.isclose(df.loc[0,"fx_obs"],fx)
    assert np.isclose(df.loc[0,"test1"],test1/test_all)
    assert np.isclose(df.loc[0,"test2"],test2/test_all)
    assert np.isclose(df.loc[0,"test3"],test3/test_all)
    assert np.isclose(df.loc[0,"fx_folded"],1-fx)

    test1 = np.exp(1)
    test2 = np.exp(0)
    test3 = np.exp(0)
    test_all = test1 + test2 + test3
    numerator = test1
    denominator = test2 + test3

    dG = -np.log(numerator/denominator)
    fx = numerator/(numerator + denominator)

    assert np.isclose(df.loc[1,"dG_obs"],dG)
    assert np.isclose(df.loc[1,"fx_obs"],fx)
    assert np.isclose(df.loc[1,"test1"],test1/test_all)
    assert np.isclose(df.loc[1,"test2"],test2/test_all)
    assert np.isclose(df.loc[1,"test3"],test3/test_all)
    assert np.isclose(df.loc[1,"fx_folded"],1-fx)

    # ------------------------------------------------------------------------
    # One observable, two not. (Use R = 1 and T = 1 to simplify math). mu_dict
    # interesting. mut_energy interesting
    ens = Ensemble(R=1)
    ens.add_species(name="test1",
                    observable=True,
                    folded=False,
                    mu_stoich={"X":1})
    ens.add_species(name="test2",
                    observable=False,
                    folded=True)
    ens.add_species(name="test3",   
                    observable=False,
                    folded=True)
    df = ens.get_obs(T=1,
                     mu_dict={"X":[0,1]},
                     mut_energy={"test2":-1})
    
    test1 = np.exp(0)
    test2 = np.exp(1)
    test3 = np.exp(0)
    test_all = test1 + test2 + test3
    numerator = test1
    denominator = test2 + test3

    dG = -np.log(numerator/denominator)
    fx = numerator/(numerator + denominator)

    assert np.isclose(df.loc[0,"dG_obs"],dG)
    assert np.isclose(df.loc[0,"fx_obs"],fx)
    assert np.isclose(df.loc[0,"test1"],test1/test_all)
    assert np.isclose(df.loc[0,"test2"],test2/test_all)
    assert np.isclose(df.loc[0,"test3"],test3/test_all)
    assert np.isclose(df.loc[0,"fx_folded"],1-fx)

    test1 = np.exp(1)
    test2 = np.exp(1)
    test3 = np.exp(0)
    test_all = test1 + test2 + test3
    numerator = test1
    denominator = test2 + test3

    dG = -np.log(numerator/denominator)
    fx = numerator/(numerator + denominator)

    assert np.isclose(df.loc[1,"dG_obs"],dG)
    assert np.isclose(df.loc[1,"fx_obs"],fx)
    assert np.isclose(df.loc[1,"test1"],test1/test_all)
    assert np.isclose(df.loc[1,"test2"],test2/test_all)
    assert np.isclose(df.loc[1,"test3"],test3/test_all)
    assert np.isclose(df.loc[1,"fx_folded"],1-fx)
    
    # ------------------------------------------------------------------------
    # One observable, two not. (Use R = 1 and T = 1 to simplify math). mu_dict
    # interesting, diff for different species. mut_energy interesting
    ens = Ensemble(R=1)
    ens.add_species(name="test1",
                    observable=True,
                    folded=True,
                    mu_stoich={"X":1})
    ens.add_species(name="test2",
                    observable=False,
                    folded=False,
                    mu_stoich={"Y":1})
    ens.add_species(name="test3",   
                    observable=False,
                    folded=False)
    df = ens.get_obs(T=1,
                     mu_dict={"X":[0,1],"Y":[0,1]},
                     mut_energy={"test2":-1})
    
    test1 = np.exp(0)
    test2 = np.exp(1)
    test3 = np.exp(0)
    test_all = test1 + test2 + test3
    numerator = test1
    denominator = test2 + test3

    dG = -np.log(numerator/denominator)
    fx = numerator/(numerator + denominator)

    assert np.isclose(df.loc[0,"dG_obs"],dG)
    assert np.isclose(df.loc[0,"fx_obs"],fx)
    assert np.isclose(df.loc[0,"test1"],test1/test_all)
    assert np.isclose(df.loc[0,"test2"],test2/test_all)
    assert np.isclose(df.loc[0,"test3"],test3/test_all)
    assert np.isclose(df.loc[0,"fx_folded"],fx)

    test1 = np.exp(1)
    test2 = np.exp(2)
    test3 = np.exp(0)
    test_all = test1 + test2 + test3
    numerator = test1
    denominator = test2 + test3

    dG = -np.log(numerator/denominator)
    fx = numerator/(numerator + denominator)

    assert np.isclose(df.loc[1,"dG_obs"],dG)
    assert np.isclose(df.loc[1,"fx_obs"],fx)
    assert np.isclose(df.loc[1,"test1"],test1/test_all)
    assert np.isclose(df.loc[1,"test2"],test2/test_all)
    assert np.isclose(df.loc[1,"test3"],test3/test_all)
    assert np.isclose(df.loc[1,"fx_folded"],fx)


    # ------------------------------------------------------------------------
    # Overflow protection

    # this dG will overflow if put in raw but should be fine if we shift. 
    max_allowed = np.log(np.finfo("d").max)

    # Make sure this should overflow if used naively
    with pytest.warns(RuntimeWarning):
        x = np.exp(max_allowed + 1)
    assert np.isinf(x)

    ens = Ensemble(R=1)
    ens.add_species(name="test1",
                    observable=True,
                    folded=True,
                    dG0=max_allowed+2)
    ens.add_species(name="test2",
                    observable=False,
                    folded=False,
                    dG0=max_allowed+1)
    
    df = ens.get_obs(T=1)
    assert np.isclose(df.loc[0,"dG_obs"],-np.log(np.exp(-1)/np.exp(0)))
    assert np.isclose(df.loc[0,"fx_obs"],np.exp(-1)/(np.exp(0) + np.exp(-1)))
    assert np.isclose(df.loc[0,"test1"],np.exp(-1)/(np.exp(0) + np.exp(-1)))
    assert np.isclose( df.loc[0,"test2"],np.exp(0)/(np.exp(0) + np.exp(-1)))
    assert np.isclose(df.loc[0,"fx_folded"],np.exp(-1)/(np.exp(0) + np.exp(-1)))


    # mu_dict argument type checking
    print("--- mu_dict ---")
    for v in variable_types["dict"]:
        print(v,type(v))
        ens = Ensemble()
        ens.add_species(name="test1")
        ens.add_species(name="test2",observable=True)
        ens.get_obs(mu_dict=v)

    for v in variable_types["not_dict"]:
        print(v,type(v))

        # none okay
        if v is None:
            continue

        ens = Ensemble()
        ens.add_species(name="test1")
        ens.add_species(name="test2",observable=True)
        with pytest.raises(ValueError):
            ens.get_obs(mu_dict=v)

    for v in variable_types["float_value_or_iter"]:
        print(v,type(v))
        
        if hasattr(v,"__iter__") and len(v) == 0:
            continue

        if issubclass(type(v),pd.DataFrame):
            continue

        mu_dict = {"X":v}
        ens = Ensemble()
        ens.add_species(name="test1")
        ens.add_species(name="test2",observable=True)
        ens.get_obs(mu_dict=mu_dict)

    not_allowed = variable_types["not_float_value_or_iter"][:]
    not_allowed.append([])
    not_allowed.append(pd.DataFrame({"X":[1,2,3]}))

    for v in not_allowed:
        print(v,type(v))
    
        mu_dict = {"X":v}
        ens = Ensemble()
        ens.add_species(name="test1")
        ens.add_species(name="test2",observable=True)
        with pytest.raises(ValueError):
            ens.get_obs(mu_dict=mu_dict)

    print("--- mut_energy ---")
    for v in [{},{"test1":1},{"test2":1},{"test1":1,"test2":1}]: 
        print(v,type(v))

        ens = Ensemble()
        ens.add_species(name="test1")
        ens.add_species(name="test2",observable=True)
        ens.get_obs(mut_energy=v)

    not_allowed = variable_types["not_dict"]
    for v in not_allowed:
        print(v,type(v))
        if v is None:
            continue

        ens = Ensemble()
        ens.add_species(name="test1")
        ens.add_species(name="test2",observable=True)
        with pytest.raises(ValueError):
            ens.get_obs(mut_energy=v)

    for v in variable_types["floats_or_coercable"]:
        print(v,type(v))
        mut_energy = {"test1":v}

        ens = Ensemble()
        ens.add_species(name="test1")
        ens.add_species(name="test2",observable=True)
        ens.get_obs(mut_energy=mut_energy)
    

    for v in variable_types["not_floats_or_coercable"]:
        print(v,type(v))
        mut_energy = {"test1":v}

        ens = Ensemble()
        ens.add_species(name="test1")
        ens.add_species(name="test2",observable=True)
        with pytest.raises(ValueError):
            ens.get_obs(mut_energy=mut_energy)


    print("--- T ---")
    for v in [1,"1",1.0]:
        print(v,type(v))

        ens = Ensemble()
        ens.add_species(name="test1")
        ens.add_species(name="test2",observable=True)
        ens.get_obs(T=v)

    not_allowed = variable_types["not_floats_or_coercable"][:]
    not_allowed.append(0.0)
    not_allowed.append(-1.0)
    for v in not_allowed:
        print(v,type(v))

        ens = Ensemble()
        ens.add_species(name="test1")
        ens.add_species(name="test2",observable=True)
        with pytest.raises(ValueError):
            ens.get_obs(T=v)
