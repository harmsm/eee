from eee.simulation.core.fitness.ff import ff_on
from eee.simulation.core.fitness.ff import ff_off
from eee.simulation.core.fitness.ff import ff_neutral
from eee.simulation.core.fitness.ff import ff_on_above
from eee.simulation.core.fitness.ff import ff_on_below

def test_ff_on():
    assert ff_on(1) == 1
    assert ff_on(0) == 0

    # test robustness to extra args
    assert ff_on(1,5,some_random_kwarg="test") == 1
    assert ff_on(0,5,some_random_kwarg="test") == 0

def test_ff_off():
    assert ff_off(1) == 0
    assert ff_off(0) == 1

    # test robustness to extra args
    assert ff_off(1,5,some_random_kwarg="test") == 0
    assert ff_off(0,5,some_random_kwarg="test") == 1

def test_ff_neutral():
    assert ff_neutral(1) == 1
    assert ff_neutral(0) == 1

    # test robustness to extra args
    assert ff_neutral(0,5,some_random_kwarg="test") == 1
    assert ff_neutral(1,5,some_random_kwarg="test") == 1

def test_ff_on_above():
    assert ff_on_above(5,threshold=4) == 1
    assert ff_on_above(5,threshold=5) == 1
    assert ff_on_above(3,threshold=4) == 0

    # test robustness to extra args
    assert ff_on_above(5,5,threshold=4,some_random_kwarg="test") == 1
    assert ff_on_above(3,5,threshold=4,some_random_kwarg="test") == 0

def test_ff_on_below():
    assert ff_on_below(5,threshold=4) == 0
    assert ff_on_below(5,threshold=5) == 1
    assert ff_on_below(3,threshold=4) == 1

    # test robustness to extra args
    assert ff_on_below(5,5,threshold=4,some_random_kwarg="test") == 0
    assert ff_on_below(3,5,threshold=4,some_random_kwarg="test") == 1
