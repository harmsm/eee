"""
Functions to analyze epistasis within the ensemble between many combinations of 
mutations.
"""

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def get_all_pairs_epistasis(ens,ddg_df,mu_dict,get_only=None):
    """
    Given an ensemble and a mutation ddG file, calculate epistasis for all 
    pairs of mutations.
    
    Parameters
    ----------
    ens : Ensemble instance
        ensemble instance holding species whose names match the column names
        in a ddg csv file
    ddg_file : pandas.DataFrame
        dataframe holding the energetic effects mutations on the species
        in the ensemble. 
    mu_dict : dict
        dictionary holding chemical potentials at which to make the calculation.
        See Ensemble docs for details.
    get_only : int, optional
        if set, take only get_only lines from the ddg file.
        
    Returns
    -------
    out_df : pandas.DataFrame
        dataframe with six columns: m1 (mutation 1), m2 (mutation 2), s1 (site 1),
        s2 (site 2), ep_mag (magnitude of the observed epistasis), ep_class 
        (class of epistasis: mag, sign, recip_sign). 
    """


    if get_only is not None:
        if get_only < len(ddg_df.index):
            idx = np.random.choice(ddg_df.index,size=get_only,replace=False)
            ddg_df = ddg_df.loc[idx,:]
    
    # Get names of molecular species
    species = ens.species 

    # Create output dictionary (source for dataframe)
    columns = ["m1","m2","s1","s2","ep_mag","ep_class"]
    out = dict([(c,[]) for c in columns])
    
    # Create status bar
    N = len(ddg_df.index)
    total = N*(N-1)//2
    with tqdm(total=total) as p:

        # Go over all entries
        for i in range(len(ddg_df.index)):
            
            # i is wildtype
            if np.sum(ddg_df.loc[ddg_df.index[i],"is_wt"]) > 0:
                p.update(1)
                continue

            # Mutational effects at site i
            mut1_dict = {}
            for s in species:
                mut1_dict[s] = ddg_df.loc[ddg_df.index[i],s]

            # Paired entries
            for j in range(i+1,len(ddg_df.index)):

                # i and j are at same site
                if ddg_df.loc[ddg_df.index[i],"site"] == ddg_df.loc[ddg_df.index[j],"site"]:
                    p.update(1)
                    continue
                
                # j is wildtype
                if np.sum(ddg_df.loc[ddg_df.index[j],"is_wt"]) > 0:
                    p.update(1)
                    continue

                # Mutational effects at site j
                mut2_dict = {}
                for s in species:
                    mut2_dict[s] = ddg_df.loc[ddg_df.index[j],s]

                # Get epistasis
                df_ep = ens.get_epistasis(mut1_dict=mut1_dict,
                                          mut2_dict=mut2_dict,
                                          mu_dict=mu_dict)

                # Get mag and class
                ep_mag = np.max(df_ep.dG_ep_mag)
                ep_class = df_ep["dG_ep_class"].iloc[np.argmax(df_ep.dG_ep_mag)]

                # Record output
                out["m1"].append(ddg_df.loc[ddg_df.index[i],"mut"])
                out["m2"].append(ddg_df.loc[ddg_df.index[j],"mut"])
                out["s1"].append(ddg_df.loc[ddg_df.index[i],"site"])
                out["s2"].append(ddg_df.loc[ddg_df.index[j],"site"])
                out["ep_mag"].append(ep_mag)
                out["ep_class"].append(ep_class)
                
                p.update(1)

    out_df = pd.DataFrame(out)

    return out_df

def summarize_epistasis(df,cutoff=None):
    """
    Summarize the epistasis in a site-by-site manner.
    
    Parameters
    ----------
    df : pandas.DataFrame
        output from get_all_pairs_epistasis
    cutoff : float, optional
        only keep epistasis whose absolute mangitude is above cutoff
    
    Returns
    -------
    summary_df : pandas.DataFrame
        dataframe with columns holding summary information for each site. The 
        columns are the mean absolute value epistatic magnitude, the mean epistatic
        magntitutde, the standard deviation of the epistaitc magnitude, the 
        number of times the site has no epistaiss, the number of time the site has
        magnitude epistasis, the number of times the site has reciprocal epistasis,
        and the number of times the site has sign epistasis. 
    """

    columns = ["site",
               "abs_mag","avg_mag","std_mag",
               "ep_none","ep_mag","ep_recip","ep_sign"]
    out = dict([(c,[]) for c in columns])

    s1_sites = np.unique(df.s1)
    s2_sites = np.unique(df.s2)
    all_sites = list(s1_sites)
    all_sites.extend(s2_sites)
    all_sites = list(set(all_sites))
    all_sites = np.array(all_sites)

    for s in all_sites:

        site_mask = np.logical_or(df.s1 == s,df.s2 == s)
        site_df = df.loc[site_mask,:]
        
        # Drop epistasis below the cutoff
        if cutoff is not None:
            above_cutoff = np.abs(site_df.ep_mag) > cutoff
            site_df = site_df.loc[above_cutoff,:]

        abs_mag = np.mean(np.abs(site_df.ep_mag))
        avg_mag = np.mean(site_df.ep_mag)
        std_mag = np.std(site_df.ep_mag)
        nothing_mask = pd.isnull(df.ep_class)

        classy_sites = site_df.loc[np.logical_not(nothing_mask),"ep_class"]
        class_bins, class_counts = np.unique(classy_sites,return_counts=True)

        count_dict = dict(zip(class_bins,class_counts))
        for c in ["mag","recip","sign"]:
            if c not in count_dict:
                count_dict[c] = 0

        out["site"].append(s)
        out["abs_mag"].append(abs_mag)
        out["avg_mag"].append(avg_mag)
        out["std_mag"].append(std_mag)
        out["ep_none"].append(np.sum(nothing_mask))
        out["ep_mag"].append(count_dict["mag"])
        out["ep_recip"].append(count_dict["recip"])
        out["ep_sign"].append(count_dict["sign"])


    summary_df = pd.DataFrame(out)

    return summary_df


