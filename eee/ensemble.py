"""
Ensemble class and helper functions.
"""

from eee._private.check.standard import check_bool
from eee._private.check.standard import check_float
from eee._private.check.eee_variables import check_mu_dict
from eee._private.check.eee_variables import check_mu_stoich
from eee._private.check.eee_variables import check_mut_energy
from eee._private.array_expander import array_expander

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
        
        self._R = R
        self._species = {}
        self._mu_list = []
        self._do_arg_checking = True
    
    def add_species(self,
                    name,
                    observable=False,
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
        """

        if name in self._species:
            err = f"A species with name {name} is already in the ensemble.\n"
            raise ValueError(err)

        # Make mu_stoich an empty dictionary if not specified
        if mu_stoich is None:
            mu_stoich = {}

        # Check sanity of inputs
        if self._do_arg_checking:
            observable = check_bool(observable,"observable")
            dG0 = check_float(dG0,"dG0")
            mu_stoich = check_mu_stoich(mu_stoich)
    
        # Record that we saw this species. 
        self._species[name] = {"observable":observable,
                               "dG0":dG0,
                               "mu_stoich":mu_stoich}

        # Record the presence of this chemical potential if we haven't seen in
        # another species. 
        if mu_stoich is not None:
            for mu in mu_stoich:
                if mu not in self._mu_list:
                    self._mu_list.append(mu)
    
    def get_species_dG(self,name,mut_energy=0,mu_dict=None):
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

        # Error check on mut_energy
        if self._do_arg_checking:
            mut_energy = check_float(value=mut_energy,
                                     variable_name="mut_energy")
            
        # Get dG0 for the species
        dG0 = self._species[name]["dG0"]

        # Perturb by a mutation
        dG0_mutated = dG0 + mut_energy

        # If no mu_dict, all chemical potentials are zero. Just perturb by 
        # mutations
        if mu_dict is None:
            return dG0_mutated
        
        # Error check on mu_dict
        if self._do_arg_checking:
            mu_dict = check_mu_dict(mu_dict)

        # Figure out if we are returning an array or single value
        mu_dict, length = array_expander(mu_dict)
        
        if length == 0:
            dG = dG0_mutated
        else:
            dG = np.ones(length,dtype=float)*dG0_mutated

        # Calculate the effect of the chemical potentials
        for m in self._species[name]["mu_stoich"]:

            # Perturb dG by chemical potential (if chemical potential is in 
            # mu_dict). 
            if m in mu_dict:
                dG -= mu_dict[m]*self._species[name]["mu_stoich"][m]

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
            potential, the fractional population of each species, and two 
            thermodynamic values:
            
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
            mu_dict = dict([(m,0.0) for m in self._mu_list])
        
        # Argument sanity checking
        if self._do_arg_checking:

            mut_energy = check_mut_energy(mut_energy)
            mu_dict = check_mu_dict(mu_dict)

            # Make sure T is a positive float
            T = check_float(value=T,
                            variable_name="T",
                            minimum_allowed=0,
                            minimum_inclusive=False)

        # If a species is not specified in the mut_energy dictionary, set the
        # mutational effect to zero
        for s in self._species:
            if s not in mut_energy:
                mut_energy[s] = 0.0

        # If a chemical potential is not specified in the mu_dict, set it to 0
        for m in self._mu_list:
            if m not in mu_dict:
                mu_dict[m] = 0.0

        # Put temperature into mu so it gets expanded by array_expander
        mu_dict["_dummy_temperature"] = T
            
        # Expand mu_dict so all values have the same length
        mu_dict, length = array_expander(mu_dict)

        # Pull the temperature back out of mu_dict and figure out beta
        beta = 1/(self._R*T)
        T = mu_dict.pop("_dummy_temperature")

        # Create pops arrays
        num_species = len(self._species)
        if length == 0:

            pops = np.zeros((1,num_species),dtype=float)

            # Make T into a length 1 array
            T = np.array([T],dtype=float)
            for m in mu_dict:
                mu_dict[m] = np.array([mu_dict[m]],dtype=float)

            length = 1
            
        else:
            pops = np.zeros((length,num_species),dtype=float)

        # Go through each species.
        for i, s in enumerate(self._species):

            # Calculate dG store -dG*beta in pops
            dG = self.get_species_dG(name=s,
                                     mut_energy=mut_energy[s],
                                     mu_dict=mu_dict)
            pops[:,i] = -dG*beta

        # Shift energies to minimize numerical errors. Shift so the species 
        # with the highest weight is close to the highest possible float. Lower
        # weight species will have weights close to zero, but we will not 
        # overflow and get a anan
        shift = np.log(np.finfo('d').max)*0.9 - np.max(pops,axis=1)
        pops = pops + shift[:,None]

        # Take exponential. 
        pops = np.exp(pops)

        # Get partition function
        Q = np.sum(pops,axis=1)
        
        # Start building an output dataframe holding the temperature and 
        # chemical potentials
        out = {}
        out["T"] = T
        for m in mu_dict:
            out[m] = mu_dict[m]
        
        # Define the observable numerator and denominator
        numerator = np.zeros(length,dtype=float)
        denominator = np.zeros(length,dtype=float)
        for i, s in enumerate(self._species):
            out[s] = pops[:,i]/Q

            if self._species[s]["observable"]:
                numerator += out[s]
            else:
                denominator += out[s]
        
        out["fx_obs"] = numerator/(denominator + numerator)

        mask = np.logical_or(denominator == 0,numerator==0)
        not_mask = np.logical_not(mask)
        
        dG_out = np.zeros(len(mask),dtype=float)
        dG_out[mask] = np.nan
        dG_out[not_mask] = -1/beta*np.log(numerator[not_mask]/denominator[not_mask])

        out["dG_obs"] = dG_out

        return pd.DataFrame(out)

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
    
    @property
    def do_arg_checking(self):
        return self._do_arg_checking

    @do_arg_checking.setter
    def do_arg_checking(self,value):
        self._do_arg_checking = value

