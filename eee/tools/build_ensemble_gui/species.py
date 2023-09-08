
from eee.tools.build_ensemble_gui.ligands import DefineLigands
from eee.tools.build_ensemble_gui.base import MetaWidget
from eee.tools.build_ensemble_gui.base import SelfRemovingWidgetContainer

import ipywidgets as widgets

class SpeciesWidget(MetaWidget):

    def __init__(self,
                 update_callback,
                 species_name="",
                 dG0=0,
                 observable=True,
                 folded=True,
                 ligands=None):
    
        super().__init__(update_callback=update_callback)

        self.build_widget(species_name=species_name,
                          dG0=dG0,
                          observable=observable,
                          folded=folded,
                          ligands=ligands)
        
    def build_widget(self,
                     species_name="",
                     dG0=0,
                     observable=True,
                     folded=True,
                     ligands=None):
        
        left_items = []
        left_items.append(widgets.Text(value=species_name,
                                       placeholder="species name",
                                       continuous_update=False))
        left_items.append(widgets.FloatText(dG0,
                                            description="dG0:",
                                            continuous_update=False))

        right_items = []
        right_items.append(widgets.Checkbox(observable,description="observable"))
        right_items.append(widgets.Checkbox(folded,description="folded"))

        # Watch these widgets
        left_items[0].observe(self._watcher)
        left_items[1].observe(self._watcher)
        right_items[0].observe(self._watcher)
        right_items[1].observe(self._watcher)
        
        top_box = widgets.HBox([widgets.VBox(left_items),
                                widgets.VBox(right_items)])
        
        # Ligand for building ligands
        self._ligands = DefineLigands(update_callback=self._watcher)
        if ligands is not None:
            for lig in ligands:
                self._ligands.add_ligand(ligand_name=lig[0],
                                         stoichiometry=lig[1])
        
        self._widget = widgets.VBox([top_box,
                                     self._ligands.widget])
        
    def get_values(self):

        if self._widget is None:
            return None

        fields = self._widget.children[0]
    
        v = {}
        v["name"] = fields.children[0].children[0].get_interact_value()
        v["dG0"] = fields.children[0].children[1].get_interact_value()
        v["observable"] = fields.children[1].children[0].get_interact_value()
        v["folded"] = fields.children[1].children[1].get_interact_value()
        
        v["ligands"] = self._ligands.get_values()

        return v
    
class DefineSpecies(SelfRemovingWidgetContainer):
    """
    Class for defining an arbitrary set of species in an ensemble.
    """
    
    def __init__(self,update_callback=None):
        """
        Create a new instance that has an "Add" button.
        """
        
        add_species_button = widgets.Button(description='Add species',
                                            disabled=False,
                                            tooltip='Add fields for another species',
                                            icon='fa-plus')
        add_species_button.on_click(self.add_species)
        
        main = widgets.VBox([add_species_button])
        super().__init__(parent_widget=main,
                         update_callback=update_callback,
                         hr_between_rows=True)

    def add_species(self,
                    button=None,
                    species_name="",
                    dG0=0,
                    observable=True,
                    folded=True,
                    ligands=None):
        """
        Add a species. Can be called as a button click callback or directly via
        the api. NOTE: No error checking is done in this function. We assume 
        this is done via the widgets in the gui. 
         
        Parameters
        ----------
        button : None or ipywidgets.Button
            The "button" argument is here so this can be called as a button
            callback, which passes the button instance as the first argument. 
        species_name : str, default=""
            name of the species
        dG0 : float, default = 0
            species dG0
        observable : bool, default=True
            whether species is observable
        folded : bool, default=True
            whether species is folded
        ligands : list-like, optional
            list of tuples defining ligands for the species. tuples should have
            the format (ligand_name,ligand_stoichiometry)
        """
        
        meta_widget = SpeciesWidget(update_callback=self._update_callback,
                                    species_name=species_name,
                                    dG0=dG0,
                                    observable=observable,
                                    folded=folded,
                                    ligands=ligands)
                
        self._add_with_remove_button(some_meta_widget=meta_widget)
        
            
        


