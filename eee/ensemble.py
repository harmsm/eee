"""
Ensemble class and helper functions
"""

import numpy as np
import pandas as pd

def _array_expander(values):
    """
    Given a list or dict of input values (that could be a mixture of arrays and
    single values), return a list or dict of float arrays where each has the
    same length. Single values are expanded to be the length of the other
    arrays. If all entries are floats, return a list or dict of floats.
    """

    # Is values a dictionary?
    if issubclass(type(values),dict):
        out = {}
        is_dict = True
    else:
        out = []
        is_dict = False

    # Does values contain an iterable?
    lengths_seen = []
    is_array = False
    for v in values:

        if is_dict:
            a = values[v]
        else:
            a = v

        if hasattr(a,"__iter__"):
            is_array = True
            lengths_seen.append(len(a))
    
    # If there was an iterable in values, expand all to be that length
    if is_array:

        # Multiple lengths seen -- throw an error
        if len(set(lengths_seen)) != 1:
            err = "All arrays must have the same length\n"
            raise ValueError(err)
        length = lengths_seen[0]

        # Go through each entry
        for v in values:

            if is_dict:
                a = values[v]
            else:
                a = v

            # If already an iterable, make sure its a float array; otherwise, 
            # create a new numpy array that is all the same value
            if hasattr(a,"__iter__"):
                new_value = np.array(a,dtype=float)
            else:
                new_value = a*np.ones(length,dtype=float)
            
            # Record results
            if is_dict:
                out[v] = new_value
            else:
                out.append(new_value)

    # If it was all floats, just return the float
    else:
        out = values
        length = 0

    return out, length

def _get_epistasis(m00,m10,m01,m11):
    """
    Get epistasis for values given in m00, m10, m01, and m11. These values 
    have some quantitative observable for wt (00), individual mutants (m01 and
    m10), and the double mutant (m11). 
    """

    # magnitude of epistasis (signed)
    mag = (m11 - m10) - (m01 - m00)
    
    # Sign of mutation 1 (will be False if mutation effect has same
    # sign in both backgrounds; True if opposite signs)
    sign1 = (m11 - m01)/(m10 - m00)
    sign1 = sign1 < 0

    # Sign of mutation 1 (will be False if mutation effect has same
    # sign in both backgrounds; True if opposite signs)
    sign2 = (m11 - m10)/(m01 - m00)
    sign2 = sign2 < 0

    # Is epistasis > 0?
    is_epistasis = np.logical_not(np.isclose(mag,0))
    
    # Some kind of sign epistasis
    is_sign = np.logical_or(sign1,sign2)

    # Magnitude if not sign
    is_mag = np.logical_not(is_sign)
    
    # Separate reciprocal from simple sign epistasis
    is_recip = np.logical_and(sign1,sign2)
    
    is_sign = np.logical_and(is_sign,np.logical_not(is_recip))

    # Filter all classes of epistasis based on magnitude
    is_mag = np.logical_and(is_mag,is_epistasis)
    is_sign = np.logical_and(is_sign,is_epistasis)
    is_recip = np.logical_and(is_recip,is_epistasis)
    
    # Record classes
    ep_class = np.array([None for _ in range(len(mag))])
    ep_class[is_mag] = "mag"
    ep_class[is_sign] = "sign"
    ep_class[is_recip] = "recip"

    return mag, sign1, sign2, ep_class

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

        # Get dG0 for the species
        dG0 = self._species[name]["dG0"]

        # Perturb by a mutation
        dG0_mutated = dG0 + mut_energy

        # If no mu_dict, all chemical potentials are zero. Just perturb by 
        # mutations
        if mu_dict is None:
            return dG0_mutated
        
        # Figure out if we are returning an array or single value
        mu_dict, length = _array_expander(mu_dict)
        if length == 0:
            dG = dG0_mutated
        else:
            dG = np.ones(length,dtype=float)*dG0_mutated

        # Calculate the effect of the chemical potentials
        for m in self._species[name]["mu_stoich"]:

            # Perturb dG by chemical potential (if chemical potential is in 
            # mu_dict). 
            if m in mu_dict:
                dG += mu_dict[m]*self._species[name]["mu_stoich"][m]

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

        # If no mutation energy dictionary specified, make one with zero for 
        # every species
        if mut_energy is None:
            mut_energy = dict([(s,0.0) for s in self._species])
        
        # If a species is not specified in the mut_energy dictionary, set the
        # mutational effect to zero
        for s in self._species:
            if s not in mut_energy:
                mut_energy[s] = 0.0

        # If no mu_dict specified, make one with 0 for every chemical potential
        if mu_dict is None:
            mu_dict = dict([(m,0.0) for m in self._mu_list])
        
        # If a chemical potential is not specified in the mu_dict, set it to 0
        for m in self._mu_list:
            if m not in mu_dict:
                mu_dict[m] = 0.0

        # Put temperature into mu so it gets expanded by _array_expander
        mu_dict["_dummy_temperature"] = T
            
        # Expand mu_dict so all values have the same length
        mu_dict, length = _array_expander(mu_dict)

        num_species = len(self._species)

        # Pull the temperature back out of mu_dict and figure out beta
        beta = -1/(self._R*T)
        T = mu_dict.pop("_dummy_temperature")

        # Create pops arrays
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

            # Calculate dG and boltzmann weight
            dG = self.get_species_dG(name=s,
                                     mut_energy=mut_energy[s],
                                     mu_dict=mu_dict)
            pops[:,i] = np.exp(dG*beta)

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
        out["dG_obs"] = 1/beta*np.log(numerator/denominator)

        return pd.DataFrame(out)


    def get_epistasis(self,
                      mut1_dict=None,
                      mut2_dict=None,
                      mut12_dict=None,
                      mu_dict=None,
                      T=298.15):

        # Calculate observables for each genotype

        df_00 = self.get_obs(mut_energy=None,
                             mu_dict=mu_dict,
                             T=T)

        df_10 = self.get_obs(mut_energy=mut1_dict,
                             mu_dict=mu_dict,
                             T=T)

        df_01 = self.get_obs(mut_energy=mut2_dict,
                             mu_dict=mu_dict,
                             T=T)

        df_11 = self.get_obs(mut_energy=mut12_dict,
                             mu_dict=mu_dict,
                             T=T)

        # Create dataframe
        columns = self._mu_list[:]
        columns.insert(0,"T")
        df = df_00.loc[:,columns]

        # Epistasis in fx_obs
        df["fx_obs_00"] = df_00.loc[:,"fx_obs"]
        df["fx_obs_10"] = df_10.loc[:,"fx_obs"]
        df["fx_obs_01"] = df_01.loc[:,"fx_obs"]
        df["fx_obs_11"] = df_11.loc[:,"fx_obs"]

        ep_mag, ep_sign1, ep_sign2, ep_class = _get_epistasis(df_00.loc[:,"fx_obs"],
                                                              df_10.loc[:,"fx_obs"],
                                                              df_01.loc[:,"fx_obs"],
                                                              df_11.loc[:,"fx_obs"])
            
        df["fx_ep_mag"] = ep_mag
        df["fx_ep_sign1"] = ep_sign1
        df["fx_ep_sign2"] = ep_sign2
        df["fx_ep_class"] = ep_class

        # Epistasis in dG_obs
        df["dG_obs_00"] = df_00.loc[:,"dG_obs"]
        df["dG_obs_10"] = df_10.loc[:,"dG_obs"]
        df["dG_obs_01"] = df_01.loc[:,"dG_obs"]
        df["dG_obs_11"] = df_11.loc[:,"dG_obs"]

        ep_mag, ep_sign1, ep_sign2, ep_class = _get_epistasis(df_00.loc[:,"dG_obs"],
                                                              df_10.loc[:,"dG_obs"],
                                                              df_01.loc[:,"dG_obs"],
                                                              df_11.loc[:,"dG_obs"])
            
        df["dG_ep_mag"] = ep_mag
        df["dG_ep_sign1"] = ep_sign1
        df["dG_ep_sign2"] = ep_sign2
        df["dG_ep_class"] = ep_class

        return df


    @property
    def species(self):
        return list(self._species.keys())
