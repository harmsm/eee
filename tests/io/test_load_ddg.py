import pytest

from eee.io.read_ddg import read_ddg

import numpy as np
import pandas as pd

def test_read_ddg(test_ddg):

    self_to_self = ["L1L","L2L","I3I","G4G"]

    raw_df = pd.read_csv(test_ddg["ddg.csv"])

    # Make sure self->self mutations are there
    assert len(raw_df) == 80
    muts = list(raw_df["mut"])
    for s in self_to_self:
        assert s in muts

    df = read_ddg(test_ddg["ddg.csv"])
    assert np.array_equal(df.columns,["site","mut","hdna","h","l2e"])

    # Make sure self->self mutations dropped
    assert len(df) == 76 
    muts = list(df["mut"])
    for s in self_to_self:
        assert s not in muts

    # Make sure sites is reasonable
    assert np.array_equal(np.unique(df["site"]),[1,2,3,4])
    
    # validate "mut" checker
    bad_df = df.loc[:,["site","hdna","h","l2e"]]
    with pytest.raises(ValueError):
        read_ddg(bad_df)

