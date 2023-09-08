import eee

from eee.tools.build_ensemble_gui.species import DefineSpecies
from eee.tools.build_ensemble_gui.titration import TitrationWidget
from eee.tools import plots

import ipywidgets as widgets
from matplotlib import pyplot as plt
import numpy as np

import inspect
import copy

class EnsembleCreator:
    """
    Create an ensemble.
    """
    
    def __init__(self,update_callback):
        
        self._update_callback = update_callback

        w = []
        w.append(widgets.HTML(value="<h2>Create an ensemble</h2>"))
        w.append(widgets.HTML(value="<hr/>"))
        
        # Species buildter
        self._species = DefineSpecies(update_callback=self._update_follower)
        w.append(self._species.widget)
        w.append(widgets.HTML(value="<hr/>"))
        
        # temperature
        ens = eee.Ensemble()
        p = inspect.signature(ens.get_obs).parameters
        default_T = p["T"].default
        
        T = widgets.BoundedFloatText(default_T,
                                     description="T:",
                                     continuous_update=False,
                                     min=np.nextafter(0, 1),
                                     max=np.finfo('d').max)
        T.observe(self._update_follower)
        
        # Gas constant
        p = inspect.signature(eee.Ensemble.__init__).parameters
        default_gas_constant = p["gas_constant"].default
        
        R = widgets.BoundedFloatText(default_gas_constant,
                                     description="R:",
                                     continuous_update=False,
                                     min=np.nextafter(0, 1),
                                     max=np.finfo('d').max)
        R.observe(self._update_follower)
        
        RT_box = widgets.HBox([T,R])
        w.append(RT_box)
        w.append(widgets.HTML(value="<hr/>"))
        
        # Titration definition
        self._titration = TitrationWidget(update_callback=self._update_follower)
        w.append(self._titration.widget)

        # Display panel
        self._out = widgets.widget_output.Output()
        w.append(self._out)
        
        self._w = w
        self._main = widgets.VBox(self._w)

        self._load_defaults()

    def _update_follower(self,*args,**kwargs):
                
        values = self.get_values()
        self._update_callback(self.get_values())
        
        # Start creating an ensemble
        ens = eee.Ensemble(gas_constant=values["gas_constant"])
        
        # Try to add species to the ensemble. If this throws an error, the
        # species info is not complete. Set good_ensemble to False
        good_ensemble = True
        for species in values["species"]:
            tmp_species = copy.deepcopy(species)
            mu_stoich = {}
            if "ligands" in tmp_species:
                for lig in tmp_species["ligands"]:
                    mu_stoich[lig[0]] = lig[1]
                tmp_species.pop("ligands")
            
            tmp_species["mu_stoich"] = mu_stoich
            try:
                ens.add_species(**tmp_species)
            except ValueError:
                good_ensemble = False
                break
            
        # Make sure we can calculate an observable from this ensemble
        if good_ensemble:
            try:
                ens.get_obs()
            except ValueError:
                good_ensemble = False
                
        # Only draw titration of ensemble if we have titration X values
        mu_dict = {}
        if good_ensemble:
            if "titration" not in values:
                good_ensemble = False
            else:
                mu_dict = values["titration"]
                
        with self._out:
            self._out.clear_output()
            if len(values["species"]) == 0:
                return
            
            if good_ensemble:

                fig, axes = plt.subplots(1,2,figsize=(12,6))
                fig, axes[0] = plots.energy_diagram(values,
                                                    fig=fig,
                                                    ax=axes[0],
                                                    context_manager=self._out)
                fig, axes[1] = plots.titration(ens=ens,
                                               mu_dict=mu_dict,
                                               fig=fig,
                                               ax=axes[1],
                                               context_manager=self._out)
            else:
                fig, ax = plt.subplots(1,figsize=(6,6))
                fig, ax = plots.energy_diagram(values,
                                               fig=fig,
                                               ax=ax,
                                               context_manager=self._out)
                
            fig.tight_layout()
            plt.show()
        
        if good_ensemble:
            with self._out:
                print(ens.to_dict())

    def get_values(self,*args,**kwargs):
        
        species = self._species.get_values()
        
        # Get all available ligands. 
        available_ligands = []
        species = self._species.get_values()
        for s in species:
            if "ligands" in s:
                for lig in s["ligands"]:
                    available_ligands.append(lig[0])
        available_ligands = list(set(available_ligands))
        available_ligands.sort()
        
        self._titration.update(current_ligands=available_ligands)
        
        temperature = self._w[4].children[0].get_interact_value()
        gas_constant = self._w[4].children[1].get_interact_value()
        ligand_titrations = self._titration.get_values()
        
        out = {"species":species,
               "gas_constant":gas_constant,
               "temperature":temperature,
               "titration":ligand_titrations}
        
        return out

    def _load_defaults(self):
        
        self._species.add_species(species_name="A",
                                  dG0=0,
                                  observable=True,
                                  folded=True)
        self._species.add_species(species_name="B",
                                  dG0=10,
                                  observable=False,
                                  folded=True,
                                  ligands=[("X",1)])
        self._species.add_species(species_name="U",
                                  dG0=10,
                                  observable=False,
                                  folded=False)

        
    def main(self):
        return self._main