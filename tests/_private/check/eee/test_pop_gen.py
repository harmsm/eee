import pytest


from eee._private.check.eee.pop_gen import check_mutation_rate
from eee._private.check.eee.pop_gen import check_population_size
from eee._private.check.eee.pop_gen import check_num_generations
from eee._private.check.eee.pop_gen import check_burn_in_generations
from eee._private.check.eee.pop_gen import check_num_mutations


def test_check_mutation_rate(variable_types):

    allowed = variable_types["floats_or_coercable"][:]
    not_allowed = variable_types["not_floats_or_coercable"][:]
    for v in allowed:
        print(v,type(v),flush=True)

        if float(v) <= 0:
            not_allowed.append(v)
            continue

        check_mutation_rate(mutation_rate=v)

    for v in not_allowed:
        print(v,type(v),flush=True)
        
        with pytest.raises(ValueError):
            check_mutation_rate(mutation_rate=v)

def test_check_population_size(variable_types):

    allowed = variable_types["ints_or_coercable"][:]
    not_allowed = variable_types["not_ints_or_coercable"][:]
    for v in allowed:
        print(v,type(v),flush=True)

        if int(v) <= 0:
            not_allowed.append(v)
            continue

        check_population_size(population_size=v)

    for v in not_allowed:
        print(v,type(v),flush=True)
        
        with pytest.raises(ValueError):
            check_population_size(population_size=v)

def test_check_num_generations(variable_types):

    allowed = variable_types["ints_or_coercable"][:]
    not_allowed = variable_types["not_ints_or_coercable"][:]
    for v in allowed:
        print(v,type(v),flush=True)

        if int(v) < 0:
            not_allowed.append(v)
            continue

        check_num_generations(num_generations=v)

    for v in not_allowed:
        print(v,type(v),flush=True)
        
        with pytest.raises(ValueError):
            check_num_generations(num_generations=v)

def test_check_burn_in_generations(variable_types):

    allowed = variable_types["ints_or_coercable"][:]
    not_allowed = variable_types["not_ints_or_coercable"][:]
    for v in allowed:
        print(v,type(v),flush=True)

        if int(v) < 0:
            not_allowed.append(v)
            continue

        check_burn_in_generations(burn_in_generations=v)

    for v in not_allowed:
        print(v,type(v),flush=True)
        
        with pytest.raises(ValueError):
            check_burn_in_generations(burn_in_generations=v)


def test_check_num_mutations(variable_types):

    allowed = variable_types["ints_or_coercable"][:]
    not_allowed = variable_types["not_ints_or_coercable"][:]
    for v in allowed:
        print(v,type(v),flush=True)

        if int(v) <= 0:
            not_allowed.append(v)
            continue

        check_num_mutations(num_mutations=v)

    for v in not_allowed:
        print(v,type(v),flush=True)
        
        with pytest.raises(ValueError):
            check_num_mutations(num_mutations=v)
    