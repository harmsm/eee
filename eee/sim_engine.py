"""
Ensemble epistasis structural analysis and simulation software.
"""

import numpy as np
import pandas as pd

from tqdm.auto import tqdm



# ------------------------------------------------------------------------------
# Analysis functions
# ------------------------------------------------------------------------------

def load_ddG(ddg_file):
    """
    Load a ddg file, enforcing the rule that all self mutations (i.e., A21A)
    have ddG = 0.

    Parameters
    ----------
    ddg_file : str
        csv file with "mut" column (formatted like A21A, Q45L, etc.) and columns
        for each species in the ensemble. The values in the species columns are
        the predicted effect of that mutation on that ensemble species. 
        
    Returns
    -------
    df : pandas.DataFrame
        pandas dataframe with columns holding 'mut', 'site' (i.e., the 21 in 
        A21P), 'is_wt' (whether or not the mutation is wildtype), and then 
        columns for the predicted ddG for each species. 
        
    Example
    -------
    Here is an example ddg_file for an ensemble with the species erg, erg-oht, 
    and erg-oht-pep.
    
    ..code-block::
    
        mut,erg,erg-oht,erg-oht-pep
        Y1A,-0.1510000000000673,-0.12999999999999545,0.4900000000000091
        Y1C,0.4880000000000564,0.17599999999993088,2.2889999999999873
        Y1D,-2.433999999999969,-2.3319999999999936,-1.6990000000000691
        Y1F,-4.645999999999958,-0.4049999999999727,3.1989999999999554
        
    
    """
    
    # Read csv file and extract sites seen
    df = pd.read_csv(ddg_file)
    df["site"] = [int(m[1:-1]) for m in list(df.mut)]

    # Find the wildtype entries (i.e., Q45Q). 
    wt_mask = np.array([m[0] == m[-1] for m in df.mut],dtype=bool)
    df["is_wt"] = wt_mask
    
    # Figure out the species columns
    columns = list(df.columns)
    columns.remove("mut")
    columns.remove("site")
    columns.remove("is_wt")
    
    # Set the ddG for wildtype to 0
    df.loc[wt_mask,columns] = 0.0
    
    return df

def get_all_pairs_epistasis(ens,ddg_file,mu_dict,get_only=None):
    """
    Given an ensemble and a mutation ddG file, calculate epistasis for all 
    pairs of mutations.
    
    Parameters
    ----------
    ens : Ensemble instance
        ensemble instance holding species whose names match the column names
        in a ddg csv file
    ddg_file : str
        file holding the energetic effects of a list of mutations on the species
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

    # Load ddG
    df = load_ddG(ddg_file)

    if get_only is not None:
        if get_only < len(df.index):
            idx = np.random.choice(df.index,size=get_only,replace=False)
            df = df.loc[idx,:]
    
    # Get names of molecular species
    species = ens.species 

    # Create output dictionary (source for dataframe)
    columns = ["m1","m2","s1","s2","ep_mag","ep_class"]
    out = dict([(c,[]) for c in columns])
    
    # Create status bar
    N = len(df.index)
    total = N*(N-1)//2
    with tqdm(total=total) as p:

        # Go over all entries
        for i in range(len(df.index)):
            
            # i is wildtype
            if np.sum(df.loc[df.index[i],"is_wt"]) > 0:
                p.update(1)
                continue

            # Mutational effects at site i
            mut1_dict = {}
            for s in species:
                mut1_dict[s] = df.loc[df.index[i],s]

            # Paired entries
            for j in range(i+1,len(df.index)):

                # i and j are at same site
                if df.loc[df.index[i],"site"] == df.loc[df.index[j],"site"]:
                    p.update(1)
                    continue
                
                # j is wildtype
                if np.sum(df.loc[df.index[j],"is_wt"]) > 0:
                    p.update(1)
                    continue

                # Mutational effects at site j
                mut2_dict = {}
                for s in species:
                    mut2_dict[s] = df.loc[df.index[j],s]

                # Get epistasis
                df_ep = ens.get_epistasis(mut1_dict=mut1_dict,
                                          mut2_dict=mut2_dict,
                                          mu_dict=mu_dict)

                # Get mag and class
                ep_mag = np.max(df_ep.dG_ep_mag)
                ep_class = df_ep["dG_ep_class"].iloc[np.argmax(df_ep.dG_ep_mag)]

                # Record output
                out["m1"].append(df.loc[df.index[i],"mut"])
                out["m2"].append(df.loc[df.index[j],"mut"])
                out["s1"].append(df.loc[df.index[i],"site"])
                out["s2"].append(df.loc[df.index[j],"site"])
                out["ep_mag"].append(ep_mag)
                out["ep_class"].append(ep_class)
                
                p.update(1)

    out_df = pd.DataFrame(out)

    return out_df

def load_bfactor(pdb_file,df,column,chain=None,out_file=None):
    """
    Load the values from a column in a dataframe into the bfactor column of a pdb
    file.
    
    Parameters
    ----------
    pdb_file : str
        pdb file to load data into
    df : pandas.DataFrame
        pandas dataframe with data. This function assumes the dataframe has a
        column called "site" that corresponds to the *sequential* number of the
        residues in the pdb file. A pdb file might have residues 51-60; the 
        dataframe could have sites 1-10, 0-9, or 51-60--as long as they 
        sequential. (It could even have skipped residues). The function does
        assume the there site for every residue in the pdb file; it will 
        choke if there is a residue in the pdb that is *not* in the dataframe.
    column : str
        column in the dataframe to use to get the data to put into the pdb file. 
        column should be a float
    chain : str, optional
        only load bfactors into the specified chain if specified
    out_file : str, optional
        output pdb file. If not specified, will write to {pdb_file}_bfactor.pdb
    """

    out = []
    
    sites_list = np.unique(df.site)
    sites_list.sort()

    resid_counter = -1
    last_resid = None
    resid_found = True
    with open(pdb_file,'r') as f:
        for line in f:
            
            if not line.startswith("ATOM"):
                out.append(line)
                continue
            
            if chain is not None:
                if line[21] != chain:
                    out.append(line)
                    continue
            
            resid = int(line[22:26])
            if resid != last_resid:
                
                last_resid = resid
                resid_counter += 1
                try:
                    site = sites_list[resid_counter]
                    resid_found = True
                except IndexError:
                    resid_found = False
            
            if resid_found:
                new_bfactor = np.array(df.loc[df.site == site,column])[0]
            else:
                new_bfactor = 0.0
            out.append(f"{line[:60]}{new_bfactor:>6.2f}{line[66:]}")


    if out_file is None:
        out_file = f"{pdb_file}_bfactor.pdb"

    f = open(out_file,"w")
    f.write("".join(out))
    f.close()

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


