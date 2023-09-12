import eee
from eee._private.check.ensemble import check_ensemble

import json
import os
import inspect

from eee.simulation.core.fitness import Fitness

#fitness_param = inspect.signature(Fitness.__init__).parameters

# def _spreadsheet_to_fitness(df,
#                             ens,
#                             default_fitness_kwargs=None,
#                             default_select_on=None,
#                             default_select_on_folded=None,
#                             default_temperature=None):
#     """
#     Load a spreadsheet and try to convert to input to a Fitness object. 
#     Rows are treated as different condition; columns as keyword parameters
#     (fitness_fcn, temperature, etc.)
#     """
    
#     # Read spreadsheet
#     df = eee.io.read_dataframe(df)
#     df_columns = set(df.columns)

#     required = set(["fitness_fcn"])
#     reserved = set(["fitness_fcn",
#                     "fitness_kwargs",
#                     "select_on",
#                     "select_on_folded",
#                     "temperature"])
    
#     # Get ligands in ensemble
#     ens = check_ensemble(ens)
#     ligands = set(ens.ligands)
        
#     # Make sure the dataframe has all required columns
#     missing_required = list(required - df_columns)
#     if len(missing_required) > 0:
#         err = "\ndataframe is missing required columns:\n"
#         for m in missing_required:
#             err += f"    {m}\n"
#         err += "\n"
#         raise ValueError(err)
    
#     # Make sure the ensemble does not have any ligands with reserved names
#     in_reserved = list(ligands.isin(reserved))
#     if len(in_reserved) > 0:
#         err = "\nensemble must not have ligands with reserved names:\n"
#         for r in in_reserved:
#             err += f"    {r}\n"
#         err += "\n"
#         raise ValueError(err)
    
#     ligand_columns = (df_columns - reserved)
#     extra_ligands = list(ligands - ligand_columns)
#     if len(extra_ligands) > 0:
#         err = "\ncolumns in the dataframe could not be interpreted as ligands.\n"
#         for lig in extra_ligands:
#             err += f"    {lig}\n"
#         err += "\n"
#         raise ValueError(err)

    
#     conditions = []
#     for idx in df.index:
        
#         condition = {"ligand_dict":{}}
#         for col in df_columns:
#             value = df.loc[idx,col]
#             if col not in ligand_columns:
#                 condition[col] = value
#             else:
#                 condition["ligand_dict"][col] = value

#         conditions.append(condition)

#     return conditions
        

# def _json_to_ensemble(json_file):

#     # Read json file
#     with open(json_file) as f:
#         calc_input = json.load(f)
    
#     # Look for "ens" key somewhere in the json output. If it's there, pull that
#     # sub-dictionary out by itself
#     key_stack = _search_for_key(calc_input,"ens")
#     if len(key_stack) == 0:
#         err = "\njson file does not have an 'ens' key\n\n"
#         raise ValueError(err)
    
#     for k in key_stack:
#         calc_input = calc_input[k]
            
#     # If specified, get gas constant out of dictionary
#     if "gas_constant" in calc_input:
#         gas_constant = calc_input.pop("gas_constant")
#     else:
#         # Get default from Ensemble class
#         gas_constant = eee.Ensemble()._gas_constant

#     # {"spreadsheet":some_file} case
#     if "spreadsheet" in calc_input:
#         return _spreadsheet_to_ensemble(df=calc_input["spreadsheet"],
#                                         gas_constant=gas_constant)
    
#     # Create ensemble from entries and validate. 
#     ens = eee.Ensemble(gas_constant=gas_constant)
#     for s in calc_input:
#         try:
#             ens.add_species(name=s,**calc_input[s])
#         except TypeError:
#             err = f"\nMangled json. Check ensemble keywords for species '{s}'\n\n"
#             raise ValueError(err)

#     return ens


# def read_ensemble(input_file):
#     """
#     Read an ensemble from a file. The file can either be json file or a 
#     spreadsheet (csv, tsv, xlsx). 

#     Parameters
#     ----------
#     input_file : str
#         input file to read

#     Notes
#     -----
#     If a the input is json, the ensemble is defined under the "ens" key. There
#     are several acceptable formats. The simplest just lists all species, where
#     the name of the species is the key and the parameters are dictionary of
#     values. Any keyword arguments to the :code:`Ensemble.add_species` method 
#     that are not defined revert to the default value for that argument in 
#     method. Some examples follow:

#     ..code-block:: json

#         {
#           "ens":{
#             "one":{"dG0":0,"mu_stoich":{"X":1},"observable":true,"folded":false},
#             "two":{"dG0":0,"mu_stoich":{"Y":1},"observable":false,"folded":false},
#           }
#         }

#     You can add the special key "gas_constant" to define the gas constant:

#     ..code-block:: json

#         {
#           "ens":{
#             "gas_constant":0.00197,
#             "one":{"dG0":0,"mu_stoich":{"X":1},"observable":true,"folded":false},
#             "two":{"dG0":0,"mu_stoich":{"Y":1},"observable":false,"folded":false},
#           }
#         }

#     You can also use the special key "spreadsheet" to point to a spreadsheet
#     file defining the ensemble. This key, if present, overrides all others. 

#     ..code-block:: json

#         {
#           "ens":{
#             "gas_constant":0.00197,
#             "spreadsheet":"ensemble.xlsx"
#           }
#         }

#     You can also embed an ensemble within more complicated json defining a 
#     simulation using the "ens" key. eee will search for the "ens" key and, if
#     present, build the ensemble from whatever is under that key. 

#     ..code-block:: json

#         {
#             "calc_type":"wf_tree_sim",
#             "calc_params":{
#                 "param_1":10,
#                 "param_2":0.01
#             },
#             "ens":{
#                 "gas_constant":0.00197,
#                 "one":{"dG0":0,"mu_stoich":{"X":1},"observable":true,"folded":false},
#                 "two":{"dG0":0,"mu_stoich":{"Y":1},"observable":false,"folded":false},
#             }
#         }

#     When reading a spreadsheet, eee treats the rows as species and the columns
#     as values. It looks for columns corresponding to the keyword arguments to 
#     :code:`Ensemble.add_species` and uses those values when adding each row to 
#     the ensemble. Omitted keywords use their default values from the method. 
#     The :code:`mu_stoich` key should *not* be used. Instead, any columns in the
#     spreadsheet that do not correspond to a keyword are treated as chemical 
#     potential stoichiometries. The following spreadsheet defines two species, 
#     s1 and s2. s1 interacts with molecule "X" with a stoichiometry of 1, s2 
#     interacts with molecule "Y" with a stoichiometry of 2.

#     +------+-----+------------+---+---+
#     | name | dG0 | observable | X | Y | 
#     +------+-----+------------+---+---+
#     | s1   | 0   | TRUE       | 1 | 0 |
#     +------+-----+------------+---+---+
#     | s2   | 5   | FALSE      | 0 | 2 |
#     +------+-----+------------+---+---+
#     """

#     input_file = f"{input_file}"
#     if not os.path.isfile(input_file):
#         err = "\ninput_file '{input_file}' is not a file\n\n"
#         raise FileNotFoundError(err)

#     # Parse as json or spreadsheet. 
#     if input_file[-5:] == ".json":
#         ens = _json_to_ensemble(input_file)
#     else:
#         ens = _spreadsheet_to_ensemble(input_file)
    
#     # Print status of loaded ensemble
#     print("\nBuilt the following ensemble\n")
#     print(ens.species_df)
#     print()

#     return ens

