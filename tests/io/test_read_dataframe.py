
import pytest

from eee.io.read_dataframe import read_dataframe

import pandas as pd
import numpy as np

import warnings

def test_read_dataframe(spreadsheets,variable_types):
    """
    Test read dataframe function.
    """

    # Read dataframe in with simple pandas call
    ref_df = pd.read_csv(spreadsheets["test.csv"])
    assert np.array_equal(ref_df.columns,["mut","hdna","h","l2e"])
    
    # Pass pd.DataFrame through and make sure it is preserved
    read_as_df = read_dataframe(ref_df)
    assert len(read_as_df) == len(ref_df)
    assert np.array_equal(read_as_df.columns,["mut","hdna","h","l2e"])
    assert read_as_df is not ref_df

    # Read in multiple formats
    for k in spreadsheets:
        print(k)

        f = spreadsheets[k]
        
        # Read file and make sure it does not throw warning.
        with warnings.catch_warnings():
            warnings.simplefilter("error")

            df = read_dataframe(f)
            assert len(df) == len(ref_df)
            assert np.array_equal(df.columns,["mut","hdna","h","l2e"])

    # Make sure dies with useful error if we send in weird inputs
    bad_inputs = variable_types["not_str"]
    for b in bad_inputs:
        print(b,type(b))

        if issubclass(type(b),pd.DataFrame):
            continue

        with pytest.raises(ValueError):
            read_dataframe(b)

    # Make sure raises file not found if a file is not passed
    with pytest.raises(FileNotFoundError):
        read_dataframe("not_really_a_file.txt")
