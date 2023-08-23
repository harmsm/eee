
from eee.simulation.core.simulation import Simulation
from eee.simulation.core.engine import exhaustive

from eee._private.check.standard import check_int
from eee._private.interface import run_cleanly


class DeepMutationalScan(Simulation):

    calc_type = "dms"

    @run_cleanly
    def run(self,
            output_directory="eee_dms",
            depth=1,
            output_file="eee_dms.csv"):
        """
        Run a deep mutational scan up to depth mutations away from wildtype. 
        
        Parameters
        ----------
        output_directory : str, default="eee_sim"
            do simulation in this output directory
        depth : int, default=1
            depth for the deep-mutational scan. 1 corresponds to all single mutants,
            2 to all double mutants, 3 to all triple, etc. WARNING: The space gets
            very large as the number of sites and number of possible mutations 
            increase. 
        output_file : str, default="eee_dms.csv"
            write results to the indicated csv file
                write the generations out every write_frequency generations. 
        """

        depth = check_int(value=depth,
                          variable_name="depth",
                          minimum_allowed=0)
        output_file = f"{output_file}"
    
        # Record the new keys
        calc_params = {}
        calc_params["depth"] = depth
        calc_params["output_file"] = output_file

        self._prepare_calc(output_directory=output_directory,
                           calc_params=calc_params)
        
        # Run and return a Wright Fisher simulation.
        exhaustive(gc=self._gc,
                   depth=depth,
                   output_file=output_file,
                   return_output=False)
        
        self._complete_calc()


    