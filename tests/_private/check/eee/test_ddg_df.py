import pytest 

from eee._private.check.eee.ddg_df import check_ddg_df

import pandas as pd

def test_check_ddg_df(variable_types):
    
    not_allowed = variable_types["everything"]
    not_allowed = [n for n in not_allowed if not issubclass(type(n),pd.DataFrame)]
    
    for v in not_allowed:
        print(v,type(v),flush=True)
        with pytest.raises(ValueError):
            check_ddg_df(v)

    df = pd.DataFrame({"test":[1,2,3]})
    check_ddg_df(df)