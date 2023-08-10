
import pytest

import numpy as np

from eee._private.array_expander import array_expander

def testarray_expander():

    out, length = array_expander({})
    assert issubclass(type(out),dict)
    assert length == 0

    out, length = array_expander([])
    assert issubclass(type(out),list)
    assert length == 0

    # Multiple lengths
    with pytest.raises(ValueError):
        array_expander([[1,2],[1]])

    # Multiple lengths
    with pytest.raises(ValueError):
        array_expander({"test":[1,2],"this":[1]})

    out, length = array_expander([[1,2],1])
    assert issubclass(type(out),list)
    assert length == 2
    assert np.array_equal(out[0],[1,2])
    assert np.array_equal(out[1],[1,1])

    out, length = array_expander({"test":[1,2],"this":1})
    assert issubclass(type(out),dict)
    assert length == 2
    assert np.array_equal(out["test"],[1,2])
    assert np.array_equal(out["this"],[1,1])
    
    with pytest.raises(TypeError):
        out, length = array_expander(1)
    
    out, length = array_expander([1,2,3])
    assert np.array_equal(out,[1,2,3])
    assert length == 0
