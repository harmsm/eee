import pytest

from eee._private.check.eee.ligand_stoich import check_ligand_stoich

def test_check_ligand_stoich(variable_types):

    # ligand_stoich argument type checking
    print("--- ligand_stoich ---")
    for v in variable_types["dict"]:
        print(v,type(v),flush=True)
        check_ligand_stoich(v)
        

    for v in variable_types["not_dict"]:
        print(v,type(v),flush=True)
        with pytest.raises(ValueError):
            check_ligand_stoich(v)