
from .wf_sim import WrightFisherSimulation

_ALLOWABLE_CALCS = {WrightFisherSimulation.calc_type:WrightFisherSimulation}
calc_list = list(_ALLOWABLE_CALCS.keys())
calc_list.sort()
