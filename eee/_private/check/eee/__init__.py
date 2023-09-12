"""
Functions for validating eee-specific arguments used throughout the codebase.
"""

from .ddg_df import check_ddg_df
from .ligand_dict import check_ligand_dict
from .ligand_stoich import check_ligand_stoich
from .mut_energy import check_mut_energy
from .temperature import check_temperature
from .wf_population import check_wf_population

from .pop_gen import check_mutation_rate
from .pop_gen import check_burn_in_generations
from .pop_gen import check_num_generations
from .pop_gen import check_num_mutations
from .pop_gen import check_population_size