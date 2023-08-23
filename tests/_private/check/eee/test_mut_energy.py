import pytest

from eee._private.check.eee.mut_energy import check_mut_energy


def test_check_mut_energy(variable_types):

    for v in [{},{"test1":1},{"test2":1},{"test1":1,"test2":1}]: 
        print(v,type(v),flush=True)
        check_mut_energy(v)

    not_allowed = variable_types["not_dict"]
    for v in not_allowed:
        print(v,type(v),flush=True)
        with pytest.raises(ValueError):
            check_mut_energy(v)

    for v in variable_types["floats_or_coercable"]:
        print(v,type(v),flush=True)
        mut_energy = {"test1":v}
        check_mut_energy(mut_energy=mut_energy)
    

    for v in variable_types["not_floats_or_coercable"]:
        print(v,type(v),flush=True)
        mut_energy = {"test1":v}
        with pytest.raises(ValueError):
            check_mut_energy(mut_energy=mut_energy)


