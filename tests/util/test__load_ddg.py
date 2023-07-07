
from eee.util._load_ddg import load_ddg

import numpy as np

def test_load_ddg(test_ddg):

    for k in test_ddg:

        df = load_ddg(test_ddg[k])
        assert len(df) == 80
        assert np.array_equal(df.columns,["mut","hdna","h","l2e","site","is_wt"])
