"""
Ensemble class and helper functions.
"""

from eee._private.check.standard import check_bool
from eee._private.check.standard import check_float
from eee._private.check.eee_variables import check_mu_dict
from eee._private.check.eee_variables import check_mu_stoich
from eee._private.check.eee_variables import check_mut_energy
from eee._private.check.eee_variables import check_T

import numpy as np
import pandas as pd

class Ensemble:
    """
    Hold a thermodynamic ensemble with an arbitrary set of macromolecular
    species whose free energy can be perturbed by mutations and the chemical 
    potentials of molecules that bind to each species. 
    
    Notes
    -----
    + This analysis assumes that the macromolecular concentration is much lower
      than the Kds for any binding reactions (i.e., that we are not in the
      stoichiometric binding regime). 

    + Each species must be assigned a dG0, the chemical potentials that perturb 
      the species, and the stoichiometry of the interaction with the molecule
      defined by the chemical potential. dG0 is the difference in free energy 
      between that species and a reference species when the chemical potentials
      are all zero. The choice of reference condition for the chemical
      potentials is arbitrary. 

    Example
    -------

    Imagine we want to model a binding reaction M + X <--> MX with a Kd of
    1E-6 M. For convenience, we set the chemical potential to be 0 at 1 M X.
    When we measure binding, we are interested in the populations of M and MX.
    We would define the system as having two species: M and MX, with MX as our
    observable. The dG0 for MX is -8.18 kcal/mol because, at 1 M X, we are
    1E6 times above our Kd (dG0 = -R*T*ln(1e6)). The stoichiometry of the
    interaction is 1:1, mu_dict["X"] = 1. (If we had 2X per M, this would be 2. 
    If we had 2M per X, this would be 0.5.)

    .. code-block:: python

        ens = Ensemble()
        ens.add_species(name="M")
        ens.add_species(name="MX",observable=True,dG0=-8.18,mu_dict={"X":1})

        df = ens.get_pops_and_obs(mu_dict={"X":np.arange(-10,11)})

    The df output will give the relative populations of M and MX as a function
    of X, the fx_obs (defined as MX/(M + MX)), and dG_obs (defined as 
    -RTln(MX/M)). 

    We can make arbitrarily complicated examples. For example, we could describe
    the following equilibrium:

    M + X + 2*Y <--> MX + 2*Y <--> MXY_2

    as: 

    .. code-block:: python

        ens = Ensemble()
        ens.add_species(name="M")
        ens.add_species(name="MX", observable=True, dG0=-8.18,mu_dict={"X":1})
        ens.add_species(name="MXY",observable=True,dG0=-16.36,mu_dict={"X":1,
                                                                       "Y":2})

        df = ens.get_pops_and_obs(mu_dict={"X":np.arange(5,16),
                                           "Y":8.18})

    This would sweep X from 5 to 15 kcal/mol, but fix Y at 8.18 kcal/mol. 
    """

    def __init__(self,R=0.001987):
        """
        Initialize the Ensemble. 

        Parameters
        ----------
        R : float, default = 0.001987
            gas constant setting energy units for this calculation. Default is 
            in kcal/mol/K.
        """
        
        # Validate gas constant
        R = check_float(value=R,
                        variable_name="R",
                        minimum_allowed=0,
                        minimum_inclusive=False)
        
        self._R = R
        self._species = {}
        
        # Lists with species and mu in stable order
        self._species_list = []
        self._mu_list = []

        # Used to avoid/minimize numerical errors in partition function 
        # calculation. 
        self._max_allowed = np.log(np.finfo('d').max)*0.01
    
    def add_species(self,
                    name,
                    observable=False,
                    folded=True,
                    dG0=0,
                    mu_stoich=None):
        """
        Add a new molecular species to the ensemble. 

        Parameters
        ----------
        name : str
            name of the species. This must be unique. 
        observable : bool, default=False
            whether or not this species is part of the observable. 
        dG0 : float, default = 0
            relative free energy of this conformation under the reference 
            conditions where all chemical potentials are defined as zero. 
        mu_stoich : dict, optional
            dictionary whose keys are strings referring to chemical potentials
            that can perturb the energy of this species and whose values 
            denote the stoichiometry relative to one macromolecule. 
        folded : bool, default=False
            whether or not this species is folded.  
        """

        if name in self._species:
            err = f"A species with name {name} is already in the ensemble.\n"
            raise ValueError(err)

        # Make mu_stoich an empty dictionary if not specified
        if mu_stoich is None:
            mu_stoich = {}

        # Check sanity of inputs
        observable = check_bool(observable,"observable")
        folded = check_bool(folded,"folded")
        dG0 = check_float(dG0,"dG0")
        mu_stoich = check_mu_stoich(mu_stoich)
    
        # Record that we saw this species. 
        self._species[name] = {"observable":observable,
                               "folded":folded,
                               "dG0":dG0,
                               "mu_stoich":mu_stoich}

        # Record the presence of this chemical potential if we haven't seen in
        # another species. 
        if mu_stoich is not None:
            for mu in mu_stoich:
                if mu not in self._mu_list:
                    self._mu_list.append(mu)

        # Stable list of species
        self._species_list.append(name)
    
    def _build_z_matrix(self,mu_dict):
        """
        Build a matrix of species energies versus conditions given the 
        conditions in mu_dict. Creates self._z_matrix, self._obs_mask, and 
        self._not_obs_mask. No error checking. Private function.
        """

        # Figure out number of species in matrix
        num_species = len(self._species)

        # Figure out number of conditions in matrix. If no mu_dict specified, 
        # single condition with all mu = 0
        if len(mu_dict) == 0:
            num_conditions = 1
            for mu in self._mu_list:
                mu_dict[mu] = np.zeros(1,dtype=float)
        else:
            num_conditions = len(mu_dict[next(iter(mu_dict))])
    
        # Local dictionary with stoich for all species and all mu. If no stoich
        # for a given species/mu, set stoich to 0. 
        mu_stoich = {}
        for species_name in self._species_list:
            mu_stoich[species_name] = {}
            for mu in mu_dict:
                if mu in self._species[species_name]["mu_stoich"]:
                    mu_stoich[species_name][mu] = self._species[species_name]["mu_stoich"][mu]
                else:
                    mu_stoich[species_name][mu] = 0

        # z_matrix holds energy of all species (i) versus conditions (j). 
        # obs_mask holds which species are observable (True) or not (False)
        self._z_matrix = np.zeros((num_species,num_conditions),dtype=float)
        self._obs_mask = np.zeros(num_species,dtype=bool)
        self._folded_mask = np.zeros(num_species,dtype=bool)

        # Go through each species...
        for i, species_name in enumerate(self._species):

            # Load reference energy for that species into all conditions
            self._z_matrix[i,:] = self._species[species_name]["dG0"]
            
            # Go through conditions
            for j in range(num_conditions):

                # Perturb dG with mu for each species under each condition
                for mu in self._mu_list:
                    self._z_matrix[i,j] -= mu_dict[mu][j]*mu_stoich[species_name][mu]

            # Update obs_mask 
            self._obs_mask[i] = self._species[species_name]["observable"]
            self._folded_mask[i] = self._species[species_name]["folded"]

        # Non-observable species
        self._not_obs_mask = np.logical_not(self._obs_mask)
        self._unfolded_mask = np.logical_not(self._folded_mask)

    def _get_weights(self,mut_energy,T):
        """
        Get Boltzmann weights for each species/condition combination given 
        mutations and current temperature. No error checking. Private. 
        mut_energy must be an array as long as the number of species; T must
        be an array as long as the number of conditions. 
        """

        # Perturb z_matrix by mut_energy and divide by RT
        beta = 1/(self._R*T)
        weights = -beta[None,:]*(self._z_matrix + mut_energy[:,None])

        # Shift so highest weight is highest allowed numerically. Low weights 
        # might underflow, but these will approach zero population anyway and 
        # can be neglected. 
        shift = self._max_allowed - np.max(weights,axis=0)
        weights = weights + shift[None,:]

        # Return boltzmann weights
        return np.exp(weights)

    def get_species_dG(self,
                       name,
                       mut_energy=0,
                       mu_dict=None):
        """
        Get the free energy of a species given some mutation energy and the
        current chemical potentials.
        
        Parameters
        ----------
        name : str
            name of species. Species must have been added using the add_species
            method.
        mut_energy : float, default = 0
            perturb the energy of the species by some mut_energy
        mu_dict : dict, optional
            dictionary of chemical potentials. keys are the names of chemical
            potentials. Values are floats or arrays of floats. Any arrays 
            specified must have the same length. If a chemical potential is not
            specified in the dictionary, its value is set to 0. 
        
        Returns
        -------
        dG : float OR numpy.ndarray
            return free energy of species. If mu_dict has arrays, return an 
            array; otherwise, return a single float value.
        """

        # See if we recognize the name
        if name not in self._species:
            err = f"species {name} not recognized. Has it been added via the\n"
            err += "add_species function?"
            raise ValueError(err)

        # Variable checking 
        mut_energy = check_float(value=mut_energy,
                                 variable_name="mut_energy")
        if mu_dict is None:
            mu_dict = {}
        mu_dict, _ = check_mu_dict(mu_dict)

        # build z matrix for calculation
        self._build_z_matrix(mu_dict)

        # Add mutation energy
        idx = self._species_list.index(name)
        dG = self._z_matrix[idx,:] + mut_energy

        # If a single condition, return a single value
        if len(dG) == 1:
            dG = dG[0]
        
        return dG

    def get_obs(self,mut_energy=None,mu_dict=None,T=298.15):
        """
        Get the population and observables given the energetic effects of 
        mutations, as well as the chemical potentials in mu_dict.
        
        Parameters
        ----------
        mut_energy : dict, optional
            dictionary holding effects of mutations on different species. Keys
            should be species names, values should be floats with mutational
            effects in energy units determined by the ensemble gas constant. 
            If a species is not in the dictionary, the mutational effect for 
            that species is set to zero. 
        mu_dict : dict, optional
            dictionary of chemical potentials. keys are the names of chemical
            potentials. Values are floats or arrays of floats. Any arrays 
            specified must have the same length. If a chemical potential is not
            specified in the dictionary, its value is set to 0. 
        T : float, default=298.15
            temperature in Kelvin. This can be an array; if so, its length must
            match the length of the arrays specified in mu_dict. 
        
        Returns
        -------
        out : pandas.DataFrame
            pandas dataframe with columns for temperature, every chemical 
            potential, the fractional population of each species, the fraction
            folded, and two thermodynamic values:
            
            fx_obs, the fractional population of the observable species: 
                ([obs1] + [obs2] + ... )/([obs1] + [obs2] + ... + [not_obs1] + [not_obs2] + ...)
            dG_obs, the observable free energy:
                -RTln(([obs1] + [obs2] + ...)/([not_obs1] + [not_obs2] + ...))
        """

        # Make sure we have enough species loaded
        if len(self._species) < 2:
            err = "Add at least two species before calculating an observable.\n"
            raise ValueError(err)
        
        # Make sure we have the necessary species loaded
        num_obs = np.sum([self._species[s]["observable"] for s in self._species])
        if num_obs < 1 or num_obs > len(self._species) - 1:
            err = "To calculate an observable, at least one species must be\n"
            err += "observable and at least one must not be observable.\n"
            raise ValueError(err)
        
        # If no mutation energy dictionary specified, make one with zero for 
        # every species
        if mut_energy is None:
            mut_energy = dict([(s,0.0) for s in self._species])

        # If no mu_dict specified, make one with 0 for every chemical potential
        if mu_dict is None:
            mu_dict = dict([(m,np.zeros(1,dtype=float)) for m in self._mu_list])
        
        # Argument sanity checking
        mut_energy = check_mut_energy(mut_energy)
        mu_dict, num_conditions = check_mu_dict(mu_dict)
        T = check_T(T,num_conditions=num_conditions)

        # Get Boltzmann weights
        self._build_z_matrix(mu_dict)
        mut_energy_array = self.mut_dict_to_array(mut_energy)
        weights = self._get_weights(mut_energy_array,T)

        # Observable and not observable weights
        obs = np.sum(weights[self._obs_mask,:],axis=0)
        not_obs = np.sum(weights[self._not_obs_mask,:],axis=0)

        # Start building an output dataframe holding the temperature and 
        # chemical potentials
        out = {}
        out["T"] = T
        for m in mu_dict:
            out[m] = mu_dict[m]
        
        # Fraction of each species
        for i, species_name in enumerate(self._species_list):
            out[species_name] = weights[i]/np.sum(weights,axis=0)
        
        # Total fraction observable. 
        out["fx_obs"] = obs/(obs + not_obs)

        # dG observable with nan checking
        nan_mask = np.logical_or(obs == 0,not_obs == 0)
        not_nan_mask = np.logical_not(nan_mask)
        
        dG_out = np.zeros(len(nan_mask),dtype=float)
        dG_out[nan_mask] = np.nan
        dG_out[not_nan_mask] = -1/(self._R*T[not_nan_mask])*np.log(obs[not_nan_mask]/not_obs[not_nan_mask])

        out["dG_obs"] = dG_out

        # Fraction folded. 
        folded = np.sum(weights[self._folded_mask,:],axis=0)
        unfolded = np.sum(weights[self._unfolded_mask,:],axis=0)

        out["fx_folded"] = folded/(folded + unfolded)

        return pd.DataFrame(out)

    def load_mu_dict(self,mu_dict={}):
        """
        Build a z-matrix given a mu_dict. This allows one to run 
        get_fx_obs_fast and get_dG_obs_fast. 

        Parameters
        ----------
        mu_dict : dict, optional
            dictionary of chemical potentials. keys are the names of chemical
            potentials. Values are floats or arrays of floats. Any arrays 
            specified must have the same length. If a chemical potential is not
            specified in the dictionary, its value is set to 0. 
        """

        mu_dict, _ = check_mu_dict(mu_dict)

        self._build_z_matrix(mu_dict)

    def mut_dict_to_array(self,mut_energy):
        """
        Convert a mut_energy dictionary to an array with the correct order  
        for _get_weights. Warning: no error checking. User is responsible for
        making sure the keys match the species loaded into the ensemble and that
        the values are floats. 

        Parameters
        ----------
        mut_energy : dict, optional
            dictionary holding effects of mutations on different species. Keys
            should be species names, values should be floats with mutational
            effects in energy units determined by the ensemble gas constant. 
            If a species is not in the dictionary, the mutational effect for 
            that species is set to zero. 

        Returns
        -------
        mut_energy_array : numpy.ndarray
            numpy array of floats where each value is the effect of the mutation
            on that species. 
        """
        
        mut_energy_array = np.zeros(len(self._species_list),dtype=float)
        for i, s in enumerate(self._species_list):
            if s in mut_energy:
                mut_energy_array[i] = mut_energy[s]

        return mut_energy_array

    def get_fx_obs_fast(self,mut_energy_array,T):
        """
        Get a numpy array with the fraction observable for the ensemble. Each 
        element is a condition in mu_dict. This only works after load_mu_dict
        has been run to create the appropriate z-matrix. Warning: no error 
        checking. 

        Parameters
        ----------
        mut_energy_array : numpy.ndarray
            numpy array of float where each value of the effect of that mutation
            on an ensemble species. Should be generated using mut_dict_to_array
        T : nump.ndarray
            numpy array of float where each value is the T at a given condition.

        Returns
        -------
        fx_obs : numpy.ndarray
            vector of fraction observable a function of the conditions in 
            mu_dict.
        fx_folded : numpy.ndarray
            vector of the fraction of the molecule folded 
        """

        weights = self._get_weights(mut_energy_array,T)
        
        obs = np.sum(weights[self._obs_mask,:],axis=0)
        not_obs = np.sum(weights[self._not_obs_mask,:],axis=0)

        folded = np.sum(weights[self._folded_mask,:],axis=0)
        unfolded = np.sum(weights[self._unfolded_mask,:],axis=0)

        return obs/(obs + not_obs), folded/(folded + unfolded)
    

    def get_dG_obs_fast(self,mut_energy_array,T):
        """
        Get a numpy array with the Dg observable for the ensemble. Each 
        element is a condition in mu_dict. This only works after load_mu_dict
        has been run to create the appropriate z-matrix. Warning: no error 
        checking. 

        Parameters
        ----------
        mut_energy_array : numpy.ndarray
            numpy array of float where each value of the effect of that mutation
            on an ensemble species. Should be generated using mut_dict_to_array
        T : nump.ndarray
            numpy array of float where each value is the T at a given condition.

        Returns
        -------
        fx_obs : numpy.ndarray
            vector of dG a function of the conditions in mu_dict. 
        fx_folded : numpy.ndarray
            vector of the fraction of the molecule folded 
        """

        weights = self._get_weights(mut_energy_array,T)
        obs = np.sum(weights[self._obs_mask,:],axis=0)
        not_obs = np.sum(weights[self._not_obs_mask,:],axis=0)

        mask = np.logical_or(obs == 0,not_obs == 0)
        not_mask = np.logical_not(mask)
        
        dG_out = np.zeros(len(mask),dtype=float)
        dG_out[mask] = np.nan
        dG_out[not_mask] = -1/(self._R*T)*np.log(obs[not_mask]/not_obs[not_mask])

        folded = np.sum(weights[self._folded_mask,:],axis=0)
        unfolded = np.sum(weights[self._unfolded_mask,:],axis=0)

        return dG_out, folded/(folded + unfolded)

    def to_dict(self):
        """
        Return a json-able dictionary describing the ensemble.
        """

        out = {"ens":{}}
        attr_to_write = ["dG0","observable","mu_stoich","folded"]
        for s in self._species:

            out["ens"][s] = {}
            for a in attr_to_write:
                out["ens"][s][a] = self._species[s][a]
    
        out["R"] = self._R

        return out
    
    
    def get_observable_function(self,obs_fcn):
        """
        Get observable functions by name. 
        
        Parameters
        ----------
        obs_fcn : str
            observable function. should be one of fx_obs or dG_obs.
        
        Returns
        -------
        fcn : function
            fast observable function that takes a mutation energy array and
            temperature array as inputs
        """

        obs_functions = {"fx_obs":self.get_fx_obs_fast,
                         "dG_obs":self.get_dG_obs_fast}
        
        bad_value = False
        if not issubclass(type(obs_fcn),str):
            bad_value = True

        if not bad_value:
            if obs_fcn not in obs_functions:
                bad_value = True

        if bad_value:
            err = f"obs_fcn ('{obs_fcn}') should be one of:\n"
            for k in obs_functions:
                err += f"    {k}\n"
            err += "\n"
            raise ValueError(err)
        
        return obs_functions[obs_fcn]


    @property
    def species(self):
        """
        Species in the ensemble.
        """
        return list(self._species.keys())
    
    @property
    def mu_list(self):
        """
        Chemical potentials in the ensemble.
        """
        return list(self._mu_list)
    