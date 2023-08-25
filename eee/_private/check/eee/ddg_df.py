"""
Validate a ddg_df dataframe.
"""

import pandas as pd

def check_ddg_df(ddg_df):
    """
    Validate a ddg_df dataframe.

    Parameters
    ----------
    ddg_df : pandas.DataFrame
        dataframe holding the energetic effects of mutations on each conformation
        in the ensemble
    
    Returns
    -------
    ddg_df : pandas.DataFrame
        validated dataframe. 
    """

    if not issubclass(type(ddg_df),pd.DataFrame):
        err = "ddg_df should be a pandas dataframe with mutational effects"
        raise ValueError(f"\n{err}\n\n")
    
    return ddg_df