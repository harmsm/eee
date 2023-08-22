from eee.evolve.fitness.ff import ff_on
from eee.evolve.fitness.ff import ff_off
from eee.evolve.fitness.ff import ff_neutral

def test_ff_on():
    assert ff_on(1) == 1
    assert ff_on(0) == 0

def test_ff_off():
    assert ff_off(1) == 0
    assert ff_off(0) == 1

def test_ff_neutral():
    assert ff_neutral(0) == 1
    assert ff_neutral(1) == 1
