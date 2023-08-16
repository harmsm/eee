"""
Helper classes for keeping track of genotypes during an evolutionary simulation.
"""
import numpy as np
import pandas as pd

import copy
import os

class Genotype:
    """
    Class to hold a genotype including the mutated sites, mutations, and the
    combined energetic effects of all mutations. Generally this will be 
    initialized by GenotypeContainer. 
    """

    def __init__(self,
                 ens,
                 ddg_dict,
                 sites=None,
                 mutations=None,
                 mutations_accumulated=None,
                 mut_energy=None):
        """
        Initialize instance.

        Parameters
        ----------
        ens : eee.Ensemble
            initialized ensemble
        ddg_dict : dict
            nested dictionary of mutational effects [site][mutation]
        sites : list, optional
            list of sites that already have mutations
        mutations : list, optional
            list of mutations to the genotype in an absolute Hamming distance 
            sense relative to wildtype. There is guaranteed to be a single 
            mutation at each site in this list. 
        mutations_accumulated : list, optional
            list of all mutations that have occurred to this genotype, in order, 
            over its history
        mut_energy : numpy.ndarray
            numpy array holding energetic effect of mutation(s) on each species.
            array should be in ensemble.species order. 
        """

        # These point to *instances* of these objects
        self._ens = ens
        self._ddg_dict = ddg_dict

        # NOTE: mutations and mutations_accumulated are shallow copies of lists
        # of strings. This is okay because all the lists do is declare which 
        # mutations are present in a genotype -- it's fine that all genotypes 
        # share individual string instances. It also makes the code way faster
        # because we do not need to copy the strings over and over. The sites
        # and mut_energy arrays are lists of numbers and numpy arrays and are
        # thus true copies. 

        # Sites is a new copy
        if sites is None:
            self._sites = []
        else:
            self._sites = copy.copy(sites)
        
        # Mutations is a new copy. 
        if mutations is None:
            self._mutations = []
        else:
            self._mutations = copy.copy(mutations)

        # Mutations is a new copy. 
        if mutations_accumulated is None:
            self._mutations_accumulated = []
        else:
            self._mutations_accumulated = copy.copy(mutations_accumulated)

        # mut_energy is a new copy. 
        if mut_energy is None:
            self._mut_energy = np.zeros(len(self._ens.species),dtype=float)
        else:
            self._mut_energy = mut_energy.copy()
            

    def copy(self):
        """
        Return a copy of the Genotype instance. This will copy the ens and
        ddg_dict objects as references, but make new instances of the sites, 
        mutations, and mut_energy attributes. 
        """
        
        return Genotype(self._ens,
                        self._ddg_dict,
                        sites=self._sites,
                        mutations=self._mutations,
                        mutations_accumulated=self._mutations_accumulated,
                        mut_energy=self._mut_energy)


    def mutate(self,site,mutation):
        """
        Introduce a mutation at a specific site. If this mutation has already
        occurred at this site, treat as a reversion. 
        """

        # Mutation to revert before adding new one
        prev_mut = None

        # If the site was already mutated, we need to mutate site back to wt
        # before mutating to new genotype
        if site in self._sites:

            # Get site of mutation in genotype
            idx = self._sites.index(site)

            # Get mutation to revert before adding new one
            prev_mut = self._mutations[idx]

            # Subtract the energetic effect of the previous mutation
            self._mut_energy -= self._ddg_dict[site][prev_mut]

            # Remove the old mutation
            self._sites.pop(idx)
            self._mutations.pop(idx)

        # If the mutation is the same as the previous mutation, treat as a
        # reversion back to the wildtype amino acid at this site. This means no
        # energetic effect of the mutation added to mut_energy and that the 
        # mutation and sites should be removed.
        if mutation != prev_mut:

            # Update energy of each species
            self._mut_energy += self._ddg_dict[site][mutation]

            # Update genotype with new site mutation and mutation made
            self._sites.append(site)
            self._mutations.append(mutation)

            # This was a mutation from the previous mutation state to the new 
            # state. Change mutation string to indicate
            final_mut_name = mutation[:]
            if prev_mut is not None:
                final_mut_name = f"{prev_mut[-1]}{mutation[1:]}"
        
        else:
            # This was a reversion, update the mutation string to indicate this
            final_mut_name = f"{mutation[-1]}{mutation[1:-1]}{mutation[0]}"
        
        # Record mutation that occurred (new site, change at existing site, 
        # reversion at site)
        self._mutations_accumulated.append(final_mut_name)


    @property
    def sites(self):
        """
        List of sites that are mutated relative to wildtype. Index synced with
        mutations.
        """
        return self._sites

    @property
    def mutations(self):
        """
        List of mutations in this genotype relative to wildtype. Index synced
        with sites.
        """
        return self._mutations
    
    @property
    def mutations_accumulated(self):
        """
        List of all mutations in the order they occurred.
        """
        return self._mutations_accumulated

    @property
    def mut_energy(self):
        """
        Current energetic effect of all mutations on the species in the 
        ensemble. This is a numpy array, with species order corresponding to 
        the ensemble.species order. 
        """
        return self._mut_energy
    
class GenotypeContainer:
    """
    Class storing a collection of genotypes. It also allows mutation of one
    genotype into another.

    Key public attributes:

        + genotypes: all genotypes seen (as Genotype instances).

        + trajectories: the trajectory taken by each genotype. Each entry
          is a list of steps taken to get to that genotype, where the steps are
          integer indexes pointing to self.genotype. A trajectory of [0,10,48]
          means the genotype started as self.genotypes[0], became
          self.genotypes[10], then finally self.genotypes[48].
          
        + mut_energies: mutational energies for the genotype.

        + fitnesses holds the absolute fitness of each genotype.
    """

    def __init__(self,fc,ddg_df):
        """
        Initialize class. 

        Parameters
        ----------
        fc : FitnessContainer
            class for calculating fitness of each genotype given its energy
        ddg_df : pandas.DataFrame
            dataframe holding the effects of each mutation on the energy of 
            each species. 
        """
        
        # Fitness and ddg information
        self._fc = fc
        self._ddg_df = ddg_df
        self._create_ddg_dict()

        # Sites and mutations for generating mutations
        self._possible_sites = list(self._ddg_dict.keys())
        self._mutations_at_sites = dict([(s,list(self._ddg_dict[s].keys()))
                                         for s in self._possible_sites])
        self._last_index = 0
        
        # Main public attributes of the class. 
        self._genotypes = {0:Genotype(self._fc.ens,self._ddg_dict)}
        self._trajectories = {0:[0]}
        self._mut_energies = {0:self._genotypes[0].mut_energy}
        self._fitnesses = {0:self._fc.fitness(self._genotypes[0].mut_energy)}

    def _create_ddg_dict(self):
        """
        Convert a ddg_df dataframe in to a dictionary of the form:
        
        ddg_dict["site"]["mut"] = np_array_with_mutant_effects_on_species
        """

        species = self._fc.ens.species
        for s in species:
            if s not in self._ddg_df.columns:
                err = f"\nspecies {s} is not in ddg_df.\n\n"
                raise ValueError(err)
            
        self._ddg_dict = {}
        for i in self._ddg_df.index:
            row = self._ddg_df.loc[i,:]
            
            site = row["site"]
            mut = row["mut"]
            if site not in self._ddg_dict:
                self._ddg_dict[site] = {}
                
            mut_dict = {}
            for s in species:
                mut_dict[s] = row[s]
            
            self._ddg_dict[site][mut] = self._fc.ens.mut_dict_to_array(mut_dict)

    
    def mutate(self,index):
        """
        Mutate the genotype with "index" to a new genotype, returning the 
        index of the new genotype.

        Parameters
        ----------
        index : int
            index of genotype to mutation. Must be an index of the
            self.genotypes array.

        Returns
        -------
        new_index : int
            index of newly generated genotype
        """

        # Sanity check
        if index not in self._genotypes:
            err = f"\nindex ({index}) is not in genotypes\n\n"
            raise IndexError(err)
        
        # Create a new genotype and mutate
        new_genotype = self._genotypes[index].copy()

        # Randomly choose a site and mutation to introduce into the genotype
        site = np.random.choice(self._possible_sites)
        mutation = np.random.choice(self._mutations_at_sites[site])

        # Introduce mutation
        new_genotype.mutate(site,mutation)
        
        # Get index for new genotype
        new_index = self._last_index + 1
        self._last_index += 1

        # Record the new genotype
        self._genotypes[new_index] = new_genotype

        # Record the trajectory taken to reach the current genotype
        new_trajectory = self._trajectories[index][:]
        new_trajectory.append(new_index)
        self._trajectories[new_index] = new_trajectory
        
        self._mut_energies[new_index] = new_genotype.mut_energy

        # Record the fitness of this genotype
        self._fitnesses[new_index] = self._fc.fitness(new_genotype.mut_energy)

        return new_index

    def dump_to_csv(self,
                    filename,
                    keep_genotypes=None):
        """
        Dump genotypes into a csv file. This appends to the filename if the
        csv already exists. The dump removes the genotypes from the object. This
        is to allow long evolutionary simulations that could potentially take 
        a large amount of memory. If run without keep_genotypes specified, the
        resulting GenotypeContainer will be empty. 
        
        Parameters
        ----------
        filename : str
            csv file to write to
        keep_genotypes : list, optional
            list of genotypes to keep. These should be keys in self.genotypes    
        """

        # Get a dataframe describing the genotypes
        df = self.df

        # Create a genotype that only has genotypes to remove
        if keep_genotypes is not None:
            to_write = np.logical_not(df["genotype"].isin(keep_genotypes))
            df = df.loc[to_write,:]

         # Write out, either appending or creating a new file
        if os.path.isfile(filename):
            df.to_csv(filename,
                      mode="a",
                      header=False,
                      index=False)
        else:
            df.to_csv(filename,
                      mode="w",
                      header=True,
                      index=False)

        # Update genotypes, trajectories, mut_energies, and fitnesses. Default
        # is to drop all genotypes. Save those in keep_genotypes. 
        genotypes = {}
        trajectories = {}
        mut_energies = {}
        fitnesses = {}
        if keep_genotypes is not None:

            for g in keep_genotypes:
                genotypes[g] = self._genotypes[g]
                trajectories[g] = self._trajectories[g]
                mut_energies[g] = self._mut_energies[g]
                fitnesses[g] = self._fitnesses[g]
            
        self._genotypes = genotypes
        self._trajectories = trajectories
        self._mut_energies = mut_energies
        self._fitnesses = fitnesses


    @property
    def df(self):
        """
        Genotypes, trajectories, energies, and fitnesses as a pandas Dataframe.
        """

        genotypes = list(self._genotypes.keys())
        mutations = ["/".join(self._genotypes[g].mutations) for g in genotypes]
        num_mutations = [len(self._genotypes[g].mutations) for g in genotypes]
        accum_mutations = ["/".join(self._genotypes[g].mutations_accumulated)
                           for g in genotypes]
        num_accum_mut = [len(self._genotypes[g].mutations_accumulated)
                         for g in genotypes]

        parent = [self._trajectories[t][-2]
                  for t in list(self._trajectories.keys())[1:]]
        parent.insert(0,pd.NA)

        out = {"genotype":genotypes,
               "mutations":mutations,
               "num_mutations":num_mutations,
               "accum_mut":accum_mutations,
               "num_accum_mut":num_accum_mut,
               "parent":parent,
               "trajectory":[self._trajectories[t] for t in self._trajectories]}

        # Get mutation energies
        mut_energy_out = {}
        for name in self._fc.ens.species:
            mut_energy_out[name] = []

        for i in self._genotypes:
            for j, name in enumerate(self._fc.ens.species):
                mut_energy_out[name].append(self._mut_energies[i][j])

        for k in mut_energy_out:
            out[f"{k}_ddg"] = mut_energy_out[k]

        out["fitness"] = [self._fitnesses[f] for f in self._fitnesses]
        
        return pd.DataFrame(out)

    @property
    def wt_sequence(self):
        """
        Wildtype protein sequence (extracted from possible mutations).
        """

        wt = []
        for s in self._possible_sites:
            wt.append(self._mutations_at_sites[s][0][0])

        return "".join(wt)

    @property
    def genotypes(self):
        """
        Dictionary of genotypes.
        """

        return self._genotypes
    
    @property
    def trajectories(self):
        """
        Dictionary of trajectories that were taken on the way to each genotype.
        """
        return self._trajectories
    
    @property
    def mut_energies(self):
        """
        Dictionary of energies for each genotype.
        """
        return self._mut_energies

    @property
    def fitnesses(self):
        """
        Dictionary of absolute fitnesses.
        """
        return self._fitnesses
    



        


    