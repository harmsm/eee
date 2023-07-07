"""
"""
from numpy import np

import copy

class Genotype:
    """
    """

    def __init__(self,
                 ens,
                 ddg_dict,
                 sites=None,
                 mutations=None,
                 mut_energy=None):
        """
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
        self._mutations_at_sites= [list(self._ddg_dict[s].keys())
                                   for s in self._possible_sites]

            
    def copy(self):
        """
        Return a copy of the Genotype instance.
        """

        # This will copy ens and ddg_dict as references, but make new instances
        # of the sites, mutations, and mut_energy attributes. 
        return Genotype(self._ens,
                        self._ddg_dict,
                        sites=self._sites,
                        mutations=self._mutations,
                        mut_energy=self._mut_energy)


    def mutate(self):
        """
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
            self._sites.remove(idx)
            self._mutations.remove(idx)

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
    

    