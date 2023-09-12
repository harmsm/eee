
import pytest

from eee._private.check.dataframe import check_dataframe

import numpy as np
import pandas as pd

def test_check_dataframe(variable_types):

    # make sure it takes variable_name and that this runs
    with pytest.raises(ValueError):
        df = check_dataframe(value="stupid")
    with pytest.raises(ValueError):
        df = check_dataframe(value="stupid",variable_name="test")
    
    # Vanilla dict
    test_dict = {"test":[1,2],"this":[3,4]}
    df = check_dataframe(value=test_dict)
    assert set(df.columns).issubset(set(["test","this"]))
    assert np.array_equal(df["test"],[1,2])
    assert np.array_equal(df["this"],[3,4])

    # Vanilla dict with variable name
    test_dict = {"test":[1,2],"this":[3,4]}
    df = check_dataframe(value=test_dict,variable_name="test")
    assert set(df.columns).issubset(set(["test","this"]))
    assert np.array_equal(df["test"],[1,2])
    assert np.array_equal(df["this"],[3,4])

    # dict with different, but compatible lengths
    test_dict = {"test":[1,2],"this":3}
    df = check_dataframe(value=test_dict)
    assert set(df.columns).issubset(set(["test","this"]))
    assert np.array_equal(df["test"],[1,2])
    assert np.array_equal(df["this"],[3,3])

    # dict with different, but compatible lengths
    test_dict = {"test":[1],"this":[3]}
    df = check_dataframe(value=test_dict)
    assert set(df.columns).issubset(set(["test","this"]))
    assert np.array_equal(df["test"],[1])
    assert np.array_equal(df["this"],[3])

    # dict with different, incompatible lengths
    test_dict = {"test":[1],"this":[3,4]}
    with pytest.raises(ValueError):
        df = check_dataframe(value=test_dict)
    
    # Sent in as df
    test_dict = {"test":[1,2],"this":[3,4]}
    test_df = pd.DataFrame(test_dict)
    df = check_dataframe(value=test_df)
    assert set(df.columns).issubset(set(["test","this"]))
    assert np.array_equal(df["test"],[1,2])
    assert np.array_equal(df["this"],[3,4])

    # non iterable
    with pytest.raises(ValueError):
        check_dataframe(value=1)

    # Iterable, but not of dicts
    with pytest.raises(ValueError):
        check_dataframe(value=[1,2])

    conditions = [{"ff":"on","temperature":298.15}]
    df = check_dataframe(value=conditions)
    assert issubclass(type(df),pd.DataFrame)
    assert len(df) == 1
    assert np.array_equal(df["ff"],["on"])
    assert np.array_equal(df["temperature"],[298.15])

    conditions = [{"ff":"on","temperature":298.15},
                  {"ff":"off"}]
    df = check_dataframe(value=conditions)
    assert issubclass(type(df),pd.DataFrame)
    assert len(df) == 2
    assert np.array_equal(df["ff"],["on","off"])
    assert np.array_equal(df["temperature"],[298.15,298.15])

    conditions = [{"ff":"on","temperature":298.15},
                  {"ff":"off","temperature":398.15}]
    df = check_dataframe(value=conditions)
    assert issubclass(type(df),pd.DataFrame)
    assert len(df) == 2
    assert np.array_equal(df["ff"],["on","off"])
    assert np.array_equal(df["temperature"],[298.15,398.15])

    # Should throw an error because T is ambiguous in last condition
    conditions = [{"ff":"on","temperature":298.15},
                  {"ff":"off","temperature":398.15},
                  {"ff":"off"}]
    with pytest.raises(ValueError):
        df = check_dataframe(value=conditions)

    # Should now work because there is only one T
    conditions = [{"ff":"on","temperature":298.15},
                  {"ff":"off"},
                  {"ff":"off"}]
    df = check_dataframe(value=conditions)
    assert issubclass(type(df),pd.DataFrame)
    assert len(df) == 3
    assert np.array_equal(df["ff"],["on","off","off"])
    assert np.array_equal(df["temperature"],[298.15,298.15,298.15])

    # Make sure bad variables are caught
    for v in variable_types:
        if issubclass(type(v),dict):
            continue
        if issubclass(type(v),pd.DataFrame):
            continue    
        print(v,type(v))
        with pytest.raises(ValueError):
            check_dataframe(value=v)