
from eee.simulation.core.simulation import Simulation

from eee.simulation.core.engine import wright_fisher

from eee._private.check.eee import check_num_generations
from eee._private.check.eee import check_mutation_rate
from eee._private.check.eee import check_population_size
from eee._private.check.standard import check_int
from eee._private.interface import run_cleanly


class WrightFisherSimulation(Simulation):

    calc_type = "wf_sim"

    @run_cleanly
    def run(self,
            output_directory="eee_sim",
            population_size=1000,
            mutation_rate=0.01,
            num_generations=100,
            write_prefix="eee_sim",
            write_frequency=1000):
        """
        Run a simulation and save files to an output directory.
        
        Parameters
        ----------
        output_directory : str, default="eee_sim"
            do simulation in this output directory
        population_size : int, default=1000
            population size for the simulation. Should be > 0.
        mutation_rate : float, default=0.01
            mutation rate for the simulation. Should be > 0.
        num_generations : int, default=100
            number of generations to run the simulation for
        write_prefix : str, default="eee_sim"
            write output files during the run with this prefix. 
        write_frequency : int, default=1000
            write the generations out every write_frequency generations. 
        """

        population_size = check_population_size(population_size)
        mutation_rate = check_mutation_rate(mutation_rate) 
        num_generations = check_num_generations(num_generations)
        write_prefix = f"{write_prefix}"

        write_frequency = check_int(value=write_frequency,
                                    variable_name="write_frequency",
                                    minimum_allowed=1)
    
        # Record the new keys
        calc_params = {}
        calc_params["population_size"] = population_size
        calc_params["mutation_rate"] = mutation_rate
        calc_params["num_generations"] = num_generations
        calc_params["write_prefix"] = write_prefix
        calc_params["write_frequency"] = write_frequency

        self._prepare_calc(output_directory=output_directory,
                           calc_params=calc_params)
        
        # Run and return a Wright Fisher simulation.
        self._gc, _ =  wright_fisher(gc=self._gc,
                                     population=population_size,
                                     mutation_rate=mutation_rate,
                                     num_generations=num_generations,
                                     write_prefix=write_prefix,
                                     write_frequency=write_frequency,
                                     rng=self._rng)
        
        self._complete_calc()


    