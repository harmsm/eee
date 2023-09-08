import eee
from eee.tools.build_ensemble_gui.base import MetaWidget

import ipywidgets as widgets
import numpy as np

import inspect

class BasicInfoWidget(MetaWidget):

    def __init__(self,
                 update_callback=None):
        
        super().__init__(update_callback=update_callback)
        self.build_widget()

    def build_widget(self):

        # temperature
        ens = eee.Ensemble()
        p = inspect.signature(ens.get_obs).parameters
        default_T = p["T"].default
        
        T = widgets.BoundedFloatText(default_T,
                                     description="T:",
                                     continuous_update=False,
                                     min=np.nextafter(0, 1),
                                     max=np.finfo('d').max)
        T.observe(self._watcher)
        
        # Gas constant
        p = inspect.signature(eee.Ensemble.__init__).parameters
        default_gas_constant = p["gas_constant"].default
        
        R = widgets.BoundedFloatText(default_gas_constant,
                                     description="R:",
                                     continuous_update=False,
                                     min=np.nextafter(0, 1),
                                     max=np.finfo('d').max)
        R.observe(self._watcher)
        
        self._widget = widgets.HBox([T,R])

    def get_values(self):
        
        temperature = self._widget.children[0].get_interact_value()
        gas_constant = self._widget.children[1].get_interact_value()

        return {"temperature":temperature,
                "gas_constant":gas_constant}
