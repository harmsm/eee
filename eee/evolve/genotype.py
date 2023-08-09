"""
Helper functions/classes for keeping track of genotypes during an evolutionary 
simulation.
"""
import numpy as np
import pandas as pd

import copy

def _create_ddg_dict(ddg_df):
    """
    Convert a ddg_df dataframe in to a dictionary of the form:
    
    ddg_dict["site"]["mut"]["species"]
    """

    ddg_dict = {}
    for i in ddg_df.index:
        row = ddg_df.loc[i,:]
        
        site = row["site"]
        mut = row["mut"]
        if site not in ddg_dict:
            ddg_dict[site] = {}
            
        ddg_dict[site][mut] = {}
        for k in row.keys()[1:-2]:
            ddg_dict[site][mut][k] = row[k]

    return ddg_dict

class Genotype:
    """
    Class to hold a genotype including the mutated sites, mutations, and the
    combined energetic effects of all mutations.
    """

    def __init__(self,
                 ens,
                 ddg_dict,
                 sites=None,
                 mutations=None,
                 mut_energy=None):
        """
        Initialize instance.

        Parameters
        ----------
        ens : eee.Ensemble
            initialized ensemble
        ddg_dict : dict
            nested dictionary of mutational effects [site][mutation][species]
        sites : list, optional
            list of sites that already have mutations
        mutations : list, optional
            list of mutations in the genotype
        mut_energy : dict, optional
            dictionary (keyed to species) holding the total energetic effects
            of all mutations
        """

        self._ens = ens
        self._ddg_dict = ddg_dict

        if sites is None:
            sites = []
        self._sites = copy.deepcopy(sites)
        
        if mutations is None:
            mutations = []
        self._mutations = copy.deepcopy(mutations)

        # Update mut_energy
        if mut_energy is not None:
            self._mut_energy = copy.deepcopy(mut_energy)
        else:
            self._mut_energy = {}
            for s in self._ens.species:
                self._mut_energy[s] = 0

        self._possible_sites = list(self._ddg_dict.keys())
        self._mutations_at_sites = dict([(s,list(self._ddg_dict[s].keys()))
                                         for s in self._possible_sites])
        
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
                        mut_energy=self._mut_energy)


    def mutate(self):
        """
        Introduce a random mutation into the genotype. A mutation of some sort
        is guaranteed to occur. This mutation may be a mutation a site where 
        a mutation has already occurred, in which case the previous mutation 
        will be reverted before the new mutation occurs. 
        """

        # Mutation to revert before adding new one
        prev_mut = None

        # Randomly choose a site to mutate
        site = np.random.choice(self._possible_sites)

        # If the site was already mutated, we need to mutate site back to wt
        # before mutating to new genotype
        if site in self._sites:

            # Get site of mutation in genotype
            idx = self._sites.index(site)

            # Get mutation to revert before adding new one
            prev_mut = self._mutations[idx]

            # Subtract the energetic effect of the previous mutation
            for species in self._ddg_dict[site][prev_mut]:
                self._mut_energy[species] -= self._ddg_dict[site][prev_mut][species]

            # Remove the old mutation
            self._sites.pop(idx)
            self._mutations.pop(idx)

        # Choose what mutation to introduce. It must be different than the 
        # previous mutation at this site. 
        while True:
            mutation = np.random.choice(self._mutations_at_sites[site])
            if mutation != prev_mut:
                break

        # Update genotype with new site mutation, mutation made, 
        # trajectory step, overall energy, and fitness
        self._sites.append(site)
        self._mutations.append(mutation)

    @property
    def sites(self):
        return self._sites

    @property
    def mutations(self):
        return self._mutations

    @property
    def mut_energy(self):
        return self._mut_energy
    
class GenotypeContainer:
    """
    Class storing a collection of genotypes. It also allows mutation of one
    genotype into another using the ensemble
    """

    def __init__(self,fc,ddg_df):
        
        self._fc = fc
        self._ddg_df = ddg_df
        self._ddg_dict = _create_ddg_dict(self._ddg_df)

        self._genotypes = [Genotype(self._fc.ens,
                                    self._ddg_dict)]
        self._trajectories = [[0]]
        self._fitnesses = [self._fc.fitness(self._genotypes[0].mut_energy)]
    
    def mutate(self,index):
        """
        Mutate the genotype with "index" to a new genotype, returning the 
        index of the new genotype.

        Parameters
        ----------
        index : int
            index of genotype to mutation. Must be an index of the self.genotypes
            array.

        Returns
        -------
        new_index : int
            index of newly generated genotype
        """

        # Sanity check
        if index > len(self._genotypes) - 1:
            err = f"index ({index}) should be less than {len(self._genotypes)}"
            raise IndexError(err)
        
        # Create a new genotype and mutate
        new_genotype = self._genotypes[index].copy()
        new_genotype.mutate()
        
        # Get index for new genotype
        new_index = len(self._genotypes)

        # Record the new genotype
        self._genotypes.append(new_genotype)

        # Record the trajectory taken to reach the current genotype
        new_trajectory = self._trajectories[index][:]
        new_trajectory.append(new_index)
        self._trajectories.append(new_trajectory)
        
        # Record the fitness of this genotype
        self._fitnesses.append(self._fc.fitness(new_genotype.mut_energy))

        return new_index

    @property
    def genotypes(self):
        return self._genotypes
    
    @property
    def trajectories(self):
        return self._trajectories
    
    @property
    def fitnesses(self):
        return self._fitnesses
    
    @property
    def df(self):

        out = {"genotype":self._genotypes,
               "trajectory":self._trajectories,
               "fitness":self._fitnesses}
        
        return pd.DataFrame(out)


    