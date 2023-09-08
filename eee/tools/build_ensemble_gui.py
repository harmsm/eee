import eee
from eee.tools import plots

from matplotlib import pyplot as plt
import numpy as np

import ipywidgets as widgets

import inspect
import copy


class SelfRemovingWidgetContainer:
    """
    Meta-widget that allows creation of new sub-widgets with a "remove"
    button. Should be subclassed before use. Minimally, the developer will want 
    to add some sort of "add_blah" method and add a "get_values" method. 
    """
    
    def __init__(self,
                 parent_widget,
                 update_callback=None,
                 hr_between_rows=False):
        """
        Create new set of widgets, each with a remove button.
        
        Parameters
        ----------
        parent_widget : ipywidgets.widget
            all new widgets are appended to this widget (accessible via 
            the widget property)
        update_callback : callable, optional
            if defined, call this function without arguments whenever a 
            widget is added or subtracted. This function should take 
            the widget values as its only argument.
        hr_between_rows : bool, default=False
            add a horizontal row between added widgets
        """
            
        self._widget = parent_widget
        self._widget_dict = {}
        self._update_callback = update_callback
        self._hr_between_rows = hr_between_rows
        
    def _append_widget(self,some_widget):
        """
        Append some_widget to the children of self._widget.
        """

        # Actually append the widget to _base_box. 
        current_widgets = list(self._widget.children)
        current_widgets.append(some_widget)
        self._widget.children = tuple(current_widgets)
    
    def _insert_widget(self,index,some_widget):
        """
        Insert some_widget to the children of self._widget at position index.
        """

        # Actually append the widget to _base_box. 
        current_widgets = list(self._widget.children)
        current_widgets.insert(index,some_widget)
        self._widget.children = tuple(current_widgets)
        
    def _pop_widget(self,index):
        """
        Remove a widget (by position) from self._widget.
        """

        current_widgets = list(self._widget.children)
        
        try:
            w = current_widgets[index]
        except IndexError:
            return None

        self._remove_widget(w)
        
    def _remove_widget(self,some_widget):
        """
        Pop a widget (by identity) from _widget.
        """
        
        current_widgets = list(self._widget.children)
        for i, c in enumerate(current_widgets):
            if c is some_widget:
                current_widgets.pop(i)
                break
        self._widget.children = tuple(current_widgets)

        
    def _add_with_remove_button(self,
                                some_widget,
                                widget_container=widgets.HBox,
                                index=-1):
        """
        Add a widget to widget with an attached remove button.
        """
        
        # Create a button to remove the widget
        remove_button = widgets.Button(description='',
                                       disabled=False,
                                       tooltip='Remove entry',
                                       icon='fa-minus',
                                       layout=widgets.Layout(width='35px'))
        
        # Remove call back will remove the widget_container we
        # create, and thus some_widget and the remove button
        remove_button.on_click(self._remove_button_callback)
        
        # We actually add some_widget within a widget_container 
        # that has both the widget and the remove button
        to_add = widget_container([remove_button,
                                   some_widget])
        if self._hr_between_rows:
            to_add = widgets.VBox([to_add,
                                   widgets.HTML("<hr/>")])
            
        
        # Record we added the widget, keying to remove_button
        self._widget_dict[remove_button] = to_add
        
        # Add to widget
        if index is None:
            self._append_widget(to_add)
        else:
            self._insert_widget(index,to_add)
            
        if self._update_callback is not None:
            self._update_callback(self.get_values())
                
    def _remove_button_callback(self,button):
        """
        Callback for a remove button.
        """

        # Figure out the widget to remove
        to_remove = self._widget_dict[button]
                
        # Remove wiget
        self._remove_widget(to_remove)
        
        # Remove it from the control dictionaries
        self._widget_dict.pop(button)
        
        # Call the removal callback for this object
        if self._update_callback is not None:
            self._update_callback(self.get_values())
        
    @property
    def widget(self):
        """
        Parent widget.
        """
        
        return self._widget
    
    @property
    def widgets(self):
        """
        List of widgets added with remove button. 
        """
        
        widgets = []
        for k in self._widget_dict:
            widgets.append(self._widget_dict[k])
        
        return widgets
    
    def get_values(self):
        """
        Redefine this in a subclass. Should return a list holding
        values extraced from the widgets. 
        """
        
        return []

    def _watcher(self,*args,**kwargs):
        """
        Attach this to a particular subwidget (say, a field)  using 'observe'
        and it will pass the values from entire collection of widgets to 
        the update callback.
        
        some_file.observe(self._watcher)
        """
        
        if self._update_callback is not None:
            self._update_callback(self.get_values())
    

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
        fields = widgets.HBox([left_field,
                               right_field])
            
        self._add_with_remove_button(some_widget=fields)
        
        
    def get_values(self):
        """
        Get the values of all ligands. Returns a list of tuples, where each 
        tuple holds the (name,stoichiometry) of the associated ligands. 
        """
        
        out = []
        for lig in self.widgets:
            ligand_name = lig.children[1].children[0].get_interact_value()
            ligand_stoich = lig.children[1].children[1].get_interact_value()
        
            out.append((ligand_name,ligand_stoich))
            
        return out


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
        
        # Lists of ligand widgets and species widgets added. This lets us access
        # widgets directly. 
        self._ligands = []
        self._species = []
        
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
        self._ligands.append(DefineLigands(update_callback=self._watcher))
        if ligands is not None:
            for lig in ligands:
                self._ligands[-1].add_ligand(ligand_name=lig[0],
                                             stoichiometry=lig[1])
        
        widget = widgets.VBox([top_box,
                               self._ligands[-1].widget])
                
        self._add_with_remove_button(some_widget=widget)
        
        # This list lets us link species to ligands list.
        self._species.append(widget)
            

    def get_values(self):
        """
        Get values for all species and return as a list of dictionaries. Each 
        list element is a species. Dictionary elements hold name, dG0, 
        observable, folded, and ligands for the species. 
        """
        
        out = []
        for i, s in enumerate(self.widgets):
        
            fields = s.children[0].children[1].children[0]
        
            v = {}
            v["name"] = fields.children[0].children[0].get_interact_value()
            v["dG0"] = fields.children[0].children[1].get_interact_value()
            v["observable"] = fields.children[1].children[0].get_interact_value()
            v["folded"] = fields.children[1].children[1].get_interact_value()
            
            v["ligands"] = self._ligands[i].get_values()

            out.append(v)
            
        return out
    
    def _remove_button_callback(self,button):
        """
        Callback for a remove button. We override the default method for the
        class so we can nuke the species and associated ligands from 
        self._species and self._ligands. 
        """

        # Figure out the widget to remove
        to_remove = self._widget_dict[button]
                
        # Remove species and ligands for this species
        idx = self._species.index(to_remove.children[0].children[1])
        self._species.pop(idx)
        self._ligands.pop(idx)

        # Call parent callback to finish 
        super()._remove_button_callback(button=button)
        

class TitrationWidget:
    """
    Widget holding subwidgets to define titration behavior.
    """
    
    def __init__(self,
                 update_callback=None,
                 min_value=0,
                 max_value=20,
                 num_steps=100):
        """
        Create the widget.

        Parameters
        ----------
        update_callback : callable, optional
            function called when a widget element changes. should take the 
            output of get_values
        min_value : float, default=0
            default minimum value
        max_value : float, default=20
            default maximum value
        num_steps : int, default=100
            default number of steps in titration
        """
        
        self._update_callback = update_callback
        
        dd = widgets.Select(options=[],
                            value=None,
                            description="Ligand:",
                            disabled=True)

        min_value = widgets.FloatText(min_value,
                                      description="min",
                                      continuous_update=False)
        max_value = widgets.FloatText(max_value,
                                      description="max",
                                      continuous_update=False)

        num_steps = widgets.BoundedIntText(num_steps,
                                           description="num steps",
                                           continuous_update=False,
                                           min=2,
                                           max=np.iinfo("i").max)

        dd.observe(self._watcher)
        min_value.observe(self._watcher)
        max_value.observe(self._watcher)
        num_steps.observe(self._watcher)

        self._widget = widgets.HBox([dd,min_value,max_value,num_steps])
        
    def _watcher(self,*args,**kwargs):
        """
        Method that wraps the callback. This can be passed to the observe method
        for sub-widgets.
        """
                
        if self._update_callback is not None:
            self._update_callback(self.get_values())
    
    def _silent_update(self,w,key,value):
        """
        Update some widget to some value without triggering the callback. 
        """
        
        w.unobserve(self._watcher)
        w.set_trait(key,value)
        w.observe(self._watcher)
    
    def update(self,current_ligands):
        """
        Update the widget programatically. Key purpose is to synchronize the
        widget with the currently defined ligands in the species. 
        """
        
        # select box
        select_box = self._widget.children[0]
        
        # Change available options
        self._silent_update(select_box,"options",current_ligands)
        
        # Synchronize value with ligands available. 
        if select_box.value not in current_ligands:
            if len(current_ligands) == 0:
                self._silent_update(select_box,"value",None)
            else:
                self._silent_update(select_box,"value",select_box.options[0])
                    
        # Decide what to do with current ligands
        if len(current_ligands) == 0:
            self._silent_update(select_box,"disabled",True)
        else:
            self._silent_update(select_box,"disabled",False)
    
    def get_values(self):
        """
        Get the titration of the ligand as a dictionary (key is ligand name,
        value is numpy array of titration values)
        """
        
        w = self._widget
        
        ligand = w.children[0].get_interact_value()
        if ligand is None:
            return {}
        
        min_value = w.children[1].get_interact_value()
        max_value = w.children[2].get_interact_value()
        num_steps = w.children[3].get_interact_value()
        
        mu_dict = {ligand:np.linspace(start=min_value,
                                      stop=max_value,
                                      num=num_steps)}
        return mu_dict
    
    @property
    def widget(self):
        return self._widget



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