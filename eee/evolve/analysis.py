"""
Scripts for analyzing the results of Wright-Fisher evolutionary simulations. 
"""

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def get_genotype_frequencies(generations,cutoff=50):
    """
    Extract genotype frequencies over simulation. 
    
    Parameters
    ----------
    generations : list
        list of dicts holding results returned by a Wright-Fisher simulation
    cutoff : int, default=50
        drop genotypes seen fewer than cutoff times over the simulation
    
    Returns
    -------
    df : pandas.DataFrame
        dataframe with genotype frequencies over the course of the simulation. 
        columns are genotypes; rows are generations. 
    """

    max_counts = {}
    population_size = None
    for g in tqdm(generations):

        if population_size is None:
            population_size = 0
            for k in g:
                population_size += g[k]
                
        for k in g:
            if k not in list(max_counts.keys()):
                max_counts[k] = g[k]
                
            if g[k] > max_counts[k]:
                max_counts[k] = g[k]
                
    keys_to_keep = []
    for k in max_counts:
        if max_counts[k] > cutoff:
            keys_to_keep.append(k)
    
    out_dict = {}
    for k in keys_to_keep:
        out_dict[k] = np.zeros(len(generations),dtype=float)
        for i in range(len(generations)):
            try:
                out_dict[k][i] = generations[i][k]/population_size
            except KeyError:
                pass
        
    return pd.DataFrame(out_dict)
            