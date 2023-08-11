
def get_num_accumulated_mutations(gc,
                                  seen,
                                  counts):
    """
    Get the total number of mutations that have accumulated (including multiple
    mutations at the same site) for the most frequent genotype in the population.
    """

    to_sort = list(zip(counts,seen))
    to_sort.sort()
    genotype = to_sort[-1][1]
    num_mutations = len(gc.genotypes[genotype].mutations_accumulated)

    return num_mutations
