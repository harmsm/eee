"""
Fitness functions for a specific observable. Any function in this file that 
starts with ff_ will be made available for evolutionary simulations. These 
functions must take the value of an observable as their first argument. They 
may also take further keyword arguments (passed in via fitness_kwargs). Whatever
occurs after ff_ will be the name of the function in things like json files 
using the function. 

def ff_example_function(value,multiply_by):
    '''
    Return the fitness of a value as that value times multiply_by. 
    '''

    return value*multiply_by

We could use this in a simulation set up as follows. This set of kwargs will
apply the function ff_example_function to "fx_obs." ff_example_function will 
multiply "fx_obs" by 5 at an iptg chemical potential of 1. 

{"fitness_fcns":["example_function"],
 "select_on": "fx_obs",
 "fitness_kwargs":{"multiply_by":5},
 "mu_dict":{"iptg":1}}
"""

def ff_on(value):
    """
    Microscopic fitness function for fx_obs. Fitness is linearly proportional
    to value. Useful for simulating selection to keep observable 'on'.
    """
    return value

def ff_off(value):
    """
    Microscopic fitness function for fx_obs. Fitness is linearly proportional
    to 1 - value. Useful for simulating selection to keep observable 'off'. 
    """
    return 1 - value

def ff_neutral(value):
    """
    Microscopic fitness function for fx_obs. Fitness is always 1.0, modeling
    no selection on fx_obs. 
    """
    return 1.0
