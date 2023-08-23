
import pandas as pd

def check_ddg_df(ddg_df):

    if not issubclass(type(ddg_df),pd.DataFrame):
        err = "ddg_df should be a pandas dataframe with mutational effects"
        raise ValueError(f"\n{err}\n\n")
    
    return ddg_df