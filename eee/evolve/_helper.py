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

class MockTqdm():
    """
    Fake tqdm progress bar so we don't have to show a status bar if we don't
    want to. Can be substituted wherever we would use tqdm (i.e.
    tqdm(range(10)) --> MockTqdm(range(10)).
    """

    def __init__(self,*args,**kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self, type, value, traceback):
        pass

    def update(self,*args,**kwargs):
        pass
