"""
Functions for manipulating ddg files.
"""
import pandas as pd
import numpy as np

def load_ddg(ddg_file):
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


