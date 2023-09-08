
from eee.tools.build_ensemble_gui.base import MetaWidget
from eee.tools.build_ensemble_gui.base import SelfRemovingWidgetContainer

import ipywidgets as widgets

class LigandWidget(MetaWidget):

    def __init__(self,
                 update_callback,
                 ligand_name,
                 stoichiometry):
        
        super().__init__(update_callback=update_callback)
        self.build_widget(ligand_name=ligand_name,
                          stoichiometry=stoichiometry)

    def build_widget(self,
                     ligand_name,
                     stoichiometry):

        # Create set of fields
        left_field = widgets.Text(value=ligand_name,
                                  placeholder="ligand name",
                                  continuous_update=False)
        right_field = widgets.BoundedFloatText(stoichiometry,
                                               description="stoichiometry",
                                               continuous_update=False,
                                               min=0)
                
        # Make sure we are monitoring both fields
        left_field.observe(self._watcher)
        right_field.observe(self._watcher)
        
        # Add with a remove button
        self._widget = widgets.HBox([left_field,
                                     right_field])
        
    def get_values(self):
        
        if self._widget is None:
            return None

        lig = self._widget
        ligand_name = lig.children[0].get_interact_value()
        ligand_stoich = lig.children[1].get_interact_value()

        return (ligand_name,ligand_stoich)
    
class DefineLigands(SelfRemovingWidgetContainer):
    """
    Gui class for defining an arbitrary number of ligands that might bind to a 
    species.
    """
    
    def __init__(self,update_callback=None):
        """
        Create a new instance with an "Add" button.
        """

        add_ligand_button = widgets.Button(description='Add ligand',
                                           disabled=False,
                                           tooltip='Add a field for a ligand',
                                           icon='fa-plus')
        add_ligand_button.on_click(self.add_ligand)
        
        parent_widget = widgets.VBox([add_ligand_button])
        super().__init__(parent_widget=parent_widget,
                         update_callback=update_callback)
                
    def add_ligand(self,
                   button=None,
                   ligand_name="",
                   stoichiometry=1):
        """
        Add a ligand. Can be called as a button click callback or directly via
        the api. NOTE: No error checking is done in this function. We assume 
        this is done via the widgets in the gui. 
         
        Parameters
        ----------
        button : None or ipywidgets.Button
            The "button" argument is here so this can be called as a button
            callback, which passes the button instance as the first argument. 
        ligand_name : str, default=""
            name of the ligand
        stoichiometry : int, default = 1
            ligand binding stoichiometry.  
        """
        
        ligand_widget = LigandWidget(update_callback=self._update_callback,
                                     ligand_name=ligand_name,
                                     stoichiometry=stoichiometry)

        self._add_with_remove_button(some_meta_widget=ligand_widget) 
    