import eee

from eee._private.check.ensemble import check_ensemble

from eee.io import load_ddg
from .simulation_container import SimulationContainer

import json
import inspect

_ALLOWABLE_CALCS = {"wright_fisher":SimulationContainer}

def _validate_calc_kwargs(calc_type,calc_class,kwargs):
    """
    Make sure the kwargs defined in the json file match the arguments for the
    calc_class. This does not check the types of the arguments (that is done
    within each class) but does make sure we have the correct argument names.
    It will generate a human-readable error message if the arguments are 
    not correct.
    """

    # Set of kwargs found
    kwargs_found = set(list(kwargs.keys()))
        
    # Get signature with all kwargs
    sig = inspect.signature(calc_class.__init__)

    # Grab arguments, separating them into those that must be defined 
    # and those with defaults. 
    required = []
    have_defaults = []
    for param in sig.parameters:
        if sig.parameters[param].default is inspect._empty:
            if param != "self":
                required.append(param)
        else:
            have_defaults.append(param)

    # Create sets of arguments in signature
    required = set(required)
    have_defaults = set(have_defaults)
    all_allowed_args = required | have_defaults

    # Look for args that are not found or extra args
    missing_required = required - required.intersection(kwargs)
    extra_args = kwargs_found - kwargs_found.intersection(all_allowed_args)
    
    # Optimistically start by assuming success
    success = True
    miss_err = ""
    extra_err = ""
    
    # Missing arguments. Create a human-readable error string
    if len(missing_required) > 0:
        success = False

        missing_required = list(missing_required)
        missing_required.sort()

        miss_err = "\nThe following required keys are not defined:\n"
        for m in missing_required:
            miss_err += f"    {m}\n"
        miss_err += "\n"

    # Extra arguments.. Create a human-readable error string
    if len(extra_args) > 0:
        success = False

        extra_args = list(extra_args)
        extra_args.sort()

        extra_err = "\nThe following keys are not allowed:\n"
        for e in extra_args:
            extra_err += f"    {e}\n"
        extra_err += "\n"

    # If we failed above, construct a human-readable error string
    if not success:

        # Start with main error
        err = "\nThe json file does not have the correct arguments for calc_type\n"
        err += f"'{calc_type}'.\n\n"
        err = err + miss_err + extra_err 
        
        name = f"{calc_class}"
        dashes = len(name)*"-"
        
        err += f"\ncalc_type '{calc_type}' details:\n\n{name}\n{dashes}\n"   
        
        # Drop whole doc string into error message
        err += f"{calc_class.__init__.__doc__}\n\n"

        raise ValueError(err)
    
    return kwargs

def load_json(json_file,use_stored_seed=False):
    """
    Load a json file describing a simulation. This file must have the 
    following top-level keys:
    
        'ens': a dictionary of species describing the ensemble.
        'mu_dict': a dictionary indicating the chemical potentials
                    over which to do the simulation, 
        'fitness_fcns': the fitness functions to apply for each of the
                        conditions
        'ddg_df': spreadsheet file with the effects of mutations on each 
                    conformation in the ensemble. 
    
    Many other keys are permitted; see the documentation.

    Parameters
    ----------
    json_file : str
        json file to load
    use_stored_seed : bool, default=False
        The 'seed' key in the json file (if present) is ignored unless
        use_stored_seed is set to True. The only time to re-use the seed 
        is to restart a simulation or reproduce it exactly for testing 
        purposes. 
    """

    # Read json file
    with open(json_file) as f:
        run = json.load(f)
    
    if "calc_type" not in run:
        err = "\njson must have 'calc_type' key in top level that defines then\n"
        err += "calculation being done.\n\n"
        raise ValueError(err)
    calc_type = run.pop("calc_type")

    if not issubclass(type(calc_type),str) or calc_type not in _ALLOWABLE_CALCS:
        err = f"\ncalc_type '{calc_type}' is not recognized. calc_type should\n"
        err += "be one of:\n"
        for a in _ALLOWABLE_CALCS:
            err += f"    {a}\n"
        err += "\n"
        raise ValueError(err)

    calc_class = _ALLOWABLE_CALCS[calc_type]
    
    # Create an ensemble from the 'ens' key, which we assume will be required
    # in every calc_class. 
    if "ens" not in run:
        err = "\njson must have 'ens' key in top level that defines the\n"
        err += "the thermodynamic ensemble.\n\n"
        raise ValueError(err)
    
    # Get gas constant
    if "R" in run["ens"]:
        R = run["ens"].pop("R")
    else:
        # Get default from Ensemble class
        R = eee.Ensemble()._R

    # Create ensemble from entries and validate. 
    ens = eee.Ensemble(R=R)
    for e in run["ens"]:
        ens.add_species(e,**run["ens"][e])
    run["ens"] = check_ensemble(ens,check_obs=True)

    # Load ddg_df here so we don't have to keep track of the file when/if we
    # start a simulation in new directory
    if "ddg_df" in run:
        run["ddg_df"] = load_ddg(run["ddg_df"])
        
    # Drop the seed unless we are requesting it to be kept. 
    if "seed" in run and not use_stored_seed:
        run.pop("seed")

    # Validate the names of the keyword arguments
    kwargs = _validate_calc_kwargs(calc_type=calc_type,
                                   calc_class=calc_class,
                                   kwargs=run)

    # Set up the calculation class. 
    return calc_class(**kwargs)

