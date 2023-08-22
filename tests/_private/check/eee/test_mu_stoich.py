import pytest

from eee._private.check.eee.mu_stoich import check_mu_stoich

def test_check_mu_stoich(variable_types):

    # mu_dict argument type checking
    print("--- mu_dict ---")
    for v in variable_types["dict"]:
        print(v,type(v),flush=True)
        check_mu_stoich(v)
        

    for v in variable_types["not_dict"]:
        print(v,type(v),flush=True)
        with pytest.raises(ValueError):
            check_mu_stoich(v)