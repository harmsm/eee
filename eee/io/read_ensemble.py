

def _search_for_key(some_dict,
                    search_key,
                    current_stack=None):
    
    if current_stack is None:
        current_stack = []
    
    for key in some_dict:
            
        if key == search_key:
            current_stack.append(key)
            return current_stack
        
        if issubclass(type(some_dict[key]),dict):
            
            current_stack.append(key)
            current_stack = _search_for_key(some_dict=some_dict[key],
                                            search_key=search_key,
                                            current_stack=current_stack)
            if current_stack[-1] != search_key:
                current_stack = current_stack[:-1]
                
    return current_stack


# def _json_to_ensemble(json_file):

#     # Read json file
#     with open(json_file) as f:
#         calc_input = json.load(f)
    
#     # Look for "ens" key somewhere in the json output. If it's there, pull it
#     # out by itself. 
#     key_stack = _search_for_key(calc_input,"ens")
#     if len(key_stack) > 0:
#         for k in key_stack:
#             calc_input = calc_input[k]
        
#     # Get gas constant
#     if "gas_constant" in calc_input["system"]["ens"]:
#         gas_constant = calc_input["system"]["ens"].pop("gas_constant")
#     else:
#         # Get default from Ensemble class
#         gas_constant = eee.Ensemble()._gas_constant

#     # Create ensemble from entries and validate. 
#     ens = eee.Ensemble(gas_constant=gas_constant)
#     for e in calc_input["system"]["ens"]:
#         ens.add_species(e,**calc_input["system"]["ens"][e])
#     calc_input["system"]["ens"] = check_ensemble(ens,check_obs=True)


# def read_ensemble(input_file):

#     if ensemble[-5:] == ".json":
        