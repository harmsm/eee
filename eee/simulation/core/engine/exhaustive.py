
from eee._private.check.standard import check_int
from eee._private.check.standard import check_bool
from eee.simulation.core import Genotype

import numpy as np
from tqdm.auto import tqdm

import math 
import itertools
import copy

def exhaustive(gc,
               depth=1,
               output_file="exhaustive.csv",
               return_output=False):
    """
    Peform a deep mutational scan up to depth mutations away from wildtype. 
    
    Parameters
    ----------
    gc : eee.simulation.core.Genotype
        initialized Genotype object with ensemble and ddg_df loaded
    depth : int, default=1
        depth for the deep-mutational scan. 1 corresponds to all single mutants,
        2 to all double mutants, 3 to all triple, etc. WARNING: The space gets
        very large as the number of sites and number of possible mutations 
        increase. 
    output_file : str, default="exhaustive.csv"
        write results to the indicated csv file
    return_output : bool, default=False
        return the resulting dataframe. if not, return None. The output dataframe
        can be huge; we recommend using False in most circumstances, then 
        reading the dataframe in later. 
    
    Returns
    -------
    out_df : pandas.DataFrame or None
        dataframe holding the mutations that occurred, their energies and 
        fitness. if return_output == False, return None.
    """

    if not issubclass(type(gc),Genotype):
        err = "\ngc must be of type Genotype\n\n"
        raise ValueError(err)

    # Operate on a copy of gc. We're about to generate a bunch of mutants
    gc = copy.deepcopy(gc)

    depth = check_int(value=depth,
                      variable_name="depth",
                      minimum_allowed=0)

    return_output = check_bool(value=return_output,
                               variable_name="return_output")

    # Get sites to mutate
    all_sites = list(gc.ddg_dict.keys())
    num_sites = len(all_sites)

    # Figure out number of mutations at each site
    site_lengths = []
    for s in all_sites:
        site_lengths.append(len(gc.ddg_dict[s]))
    site_lengths = np.array(site_lengths)

    # Figure out number of calculations required to go from 1->depth exhaustive
    # mutations. 
    total_calcs = 0

    # Same number of mutations at each site, single loop 
    if len(np.unique(site_lengths)) == 1:
        N = site_lengths[0]
        for L in range(1,depth+1):
            total_calcs += math.comb(num_sites,L)*(N**L)

    # Different sites have different numbers of mutations -- brute force it
    else:
        for L in range(1,depth+1):
            for sites in itertools.combinations(all_sites,L):
                total_calcs += np.prod(site_lengths[np.array(sites)-1])

    # Store indexes of previous backgrounds in which to do mutations
    existing_genotype_dict = {tuple([]):0}

    pbar = tqdm(total=total_calcs)

    with pbar:

        # Go over all mutation dimensionality from 1 to depth
        for L in range(1,depth+1):

            # Create all possible combinations of sites with dimensionality L
            for sites in itertools.combinations(all_sites,L):

                # Create all possible mutations at these sites
                for mutations in itertools.product(*[gc.ddg_dict[i] for i in sites]):

                    bg = existing_genotype_dict[mutations[:-1]]
                    gc.mutate(bg,site=sites[-1],mutation=mutations[-1])
                    existing_genotype_dict[mutations] = len(gc.genotypes) - 1
                    
                    pbar.update(n=1)

    df = gc.df
    
    # Drop extra columns that really only matter for evolutionary trajectories
    all_columns = set(df.columns)
    to_drop = set(["accum_mut","num_accum_mut","parent","trajectory"])
    to_drop = list(to_drop.intersection(all_columns))
    df = df.drop(columns=to_drop)

    if not output_file is None:
        print("Writing output",flush=True)
        df.to_csv(output_file)

    out = None
    if return_output:
        out = df

    return out
    