
from eee.evolve.fitness import ff_on
from eee.evolve.fitness import ff_off
from eee.evolve.fitness import _fitness_function
from eee.evolve.fitness import fitness_function
from eee.evolve.fitness import FitnessContainer

def test_ff_on():

    assert ff_on(1) == 1
    assert ff_on(0) == 0


def test_ff_off():

    assert ff_off(1) == 0
    assert ff_off(0) == 1


def test__fitness_function():
    pass

def test_fitness_function():
    pass

def test_FitnessContainer():
    pass