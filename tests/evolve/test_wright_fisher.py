import pytest

from eee.evolve.wright_fisher import wright_fisher
from eee.evolve.genotype import GenotypeContainer

import numpy as np

import time

def test_wright_fisher(ens_test_data,ens_with_fitness,variable_types):

    def _count_pop(generation):
        return sum([generation[s] for s in generation])

    fc = ens_test_data["fc"]
    ddg_df = ens_test_data["ddg_df"]

    # --------------------------------------------------------------------------
    # Test three ways of setting the population (single number, dictionary, list)

    gc = GenotypeContainer(fc=fc,ddg_df=ddg_df)
    gc, generations = wright_fisher(gc,
                                    population=10,
                                    mutation_rate=0.1,
                                    num_generations=2)
    
    assert len(generations) == 2
    assert _count_pop(generations[0]) == 10
    assert _count_pop(generations[1]) == 10
    assert len(generations[0]) == 1
    assert generations[0][0] == 10

    gc = GenotypeContainer(fc=fc,ddg_df=ddg_df)
    gc, generations = wright_fisher(gc,
                                    population={0:10},
                                    mutation_rate=0.1,
                                    num_generations=2)
    
    assert len(generations) == 2
    assert _count_pop(generations[0]) == 10
    assert _count_pop(generations[1]) == 10
    assert len(generations[0]) == 1
    assert generations[0][0] == 10


    gc = GenotypeContainer(fc=fc,ddg_df=ddg_df)
    gc, generations = wright_fisher(gc,
                                    population=np.zeros(10,dtype=int),
                                    mutation_rate=0.1,
                                    num_generations=2)
    
    assert len(generations) == 2
    assert _count_pop(generations[0]) == 10
    assert _count_pop(generations[1]) == 10
    assert len(generations[0]) == 1
    assert generations[0][0] == 10 


    # --------------------------------------------------------------------------
    # Mutation rate

    gc = GenotypeContainer(fc=fc,ddg_df=ddg_df)
    gc, generations = wright_fisher(gc,
                                    population=10,
                                    mutation_rate=1,
                                    num_generations=2)
    high_mut_rate_genotypes = len(gc.genotypes)

    gc = GenotypeContainer(fc=fc,ddg_df=ddg_df)
    gc, generations = wright_fisher(gc,
                                    population=10,
                                    mutation_rate=0.0001,
                                    num_generations=2)
    low_mut_rate_genotypes = len(gc.genotypes)

    assert high_mut_rate_genotypes > low_mut_rate_genotypes

    # --------------------------------------------------------------------------
    # Num generations

    gc = GenotypeContainer(fc=fc,ddg_df=ddg_df)
    gc, generations = wright_fisher(gc,
                                    population=10,
                                    mutation_rate=0.1,
                                    num_generations=10)
    assert len(generations) == 10

    # --------------------------------------------------------------------------
    # num_mutations

    # 1 mutation
    gc = GenotypeContainer(fc=fc,ddg_df=ddg_df)
    gc, generations = wright_fisher(gc,
                                    population=100,
                                    mutation_rate=0.01,
                                    num_generations=1000,
                                    num_mutations=1)
    assert len(generations) < 1000
    to_sort = []
    keys = list(generations[-1].keys())
    for k in keys:
        to_sort.append((generations[-1][k],k))
    to_sort.sort()
    assert len(gc.genotypes[to_sort[-1][1]].mutations_accumulated) >= 1

    # 2 mutations
    gc = GenotypeContainer(fc=fc,ddg_df=ddg_df)
    gc, generations = wright_fisher(gc,
                                    population=100,
                                    mutation_rate=0.01,
                                    num_generations=1000,
                                    num_mutations=2)
    assert len(generations) < 1000
    to_sort = []
    keys = list(generations[-1].keys())
    for k in keys:
        to_sort.append((generations[-1][k],k))
    to_sort.sort()
    assert len(gc.genotypes[to_sort[-1][1]].mutations_accumulated) >= 2

    # 3 mutations
    gc = GenotypeContainer(fc=fc,ddg_df=ddg_df)
    gc, generations = wright_fisher(gc,
                                    population=100,
                                    mutation_rate=0.01,
                                    num_generations=1000,
                                    num_mutations=3)
    assert len(generations) < 1000
    to_sort = []
    keys = list(generations[-1].keys())
    for k in keys:
        to_sort.append((generations[-1][k],k))
    to_sort.sort()
    assert len(gc.genotypes[to_sort[-1][1]].mutations_accumulated) >= 3


    # Too short to complete 50 mutations -- should run 10 generations and warn
    gc = GenotypeContainer(fc=fc,ddg_df=ddg_df)
    with pytest.warns():
        gc, generations = wright_fisher(gc,
                                        population=10,
                                        mutation_rate=0.001,
                                        num_generations=10,
                                        num_mutations=100)
    assert len(generations) == 10


    # --------------------------------------------------------------------------
    # disable_status_bar

    # this is not a real test, but at least will test ability for the code to 
    # run without crashing when this is set. 

    gc = GenotypeContainer(fc=fc,ddg_df=ddg_df)
    gc, generations = wright_fisher(gc,
                                    population=10,
                                    mutation_rate=0.1,
                                    num_generations=10,
                                    disable_status_bar=True)


    # --------------------------------------------------------------------------
    # Check the sorts of mutations being generated

    gc = GenotypeContainer(fc=fc,ddg_df=ddg_df)
    gc, generations = wright_fisher(gc,
                                    population=10,
                                    mutation_rate=1,
                                    num_generations=2)
    high_mut_rate_genotypes = len(gc.genotypes)

    gc = GenotypeContainer(fc=fc,ddg_df=ddg_df)
    gc, generations = wright_fisher(gc,
                                    population=10,
                                    mutation_rate=0.0001,
                                    num_generations=2)
    low_mut_rate_genotypes = len(gc.genotypes)

    assert high_mut_rate_genotypes > low_mut_rate_genotypes


    # --------------------------------------------------------------------------
    # Check variables

    gc = GenotypeContainer(fc=fc,ddg_df=ddg_df)

    not_allowed = variable_types["not_ints_or_coercable"]
    for v in not_allowed:
        print(v,type(v))

        with pytest.raises(ValueError):
            gc, generations = wright_fisher(gc,
                                            population=v,
                                            mutation_rate=0.1,
                                            num_generations=2)

    not_allowed = variable_types["everything"]
    for v in not_allowed:
        print(v,type(v))

        with pytest.raises(ValueError):
            gc, generations = wright_fisher(gc=v,
                                            population=10,
                                            mutation_rate=0.1,
                                            num_generations=2)


    

    # --------------------------------------------------------------------------
    # Run a short calculation with a dataset that has three genotypes, one of
    # which is very fit, one that is very unfit, and wildtype which is in the 
    # middle

    gc = ens_with_fitness["gc"]

    gc, generations = wright_fisher(gc=gc,
                                    mutation_rate=0.001,
                                    population=10000,
                                    num_generations=1000)
    
    genotypes = {}
    for gen in generations:
        for g in gen:
            mut_genotype = "".join(gc.genotypes[g].mutations)
            if mut_genotype not in genotypes:
                genotypes[mut_genotype] = 0
            
            genotypes[mut_genotype] += gen[g]
            
    assert genotypes["A1P"] < 10000
    assert genotypes[""] > genotypes["A1P"]
    assert genotypes[""] < genotypes["A1V"]
    assert genotypes["A1V"] > 9000000


