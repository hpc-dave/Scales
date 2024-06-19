from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool
from pathos.helpers import cpu_count
import time
from typing import Callable


class Sweeper:
    def __init__(self,
                 parameter_range,
                 resolution,
                 fitness_func: Callable,
                 pre_process=None,
                 post_process=None,
                 parallelism=None,
                 save_all_results: bool = None):
        r"""
        Initializes the object and sanitizes input

        Parameters
        ----------
        parameter_range: list
            list of ranges for the parameters of interest
        resolution: list
            list of resolutions for each parameter range
        fitness_func: Callable
            function to evaluate the fitness of the solution with signature (Sweeper, list) -> float
        pre_process: Callable
            function to be executed prior to the parameter sweep
        post_process: Callable
            function to be executed after the parameter sweep
        parallelism
            execution in parallel, can be an integer to provide a number of workers or boolean and the number
            of workers will be determined automatically
        save_all_results: bool
            flag, if all results should be stored and returned
        """
        # input sanitation
        self.parameter_range = []
        self.resolution = []
        self.best_fit = None
        self.best_fitness = 0.
        self.elapsed_time = 0.
        self.save_all_results = save_all_results
        if not (isinstance(parameter_range, tuple) or isinstance(parameter_range, list)):
            raise TypeError('The parameter range has to be provided as list or tuple')
        for e in parameter_range:
            if isinstance(e, tuple) or isinstance(e, list):
                if len(e) != 2:
                    raise ValueError(f'Exactly 2 values are required for a range, received {len(e)}!')
                if not isinstance(e[0], float) and not isinstance(e[1], float)\
                   and not isinstance(e[0], int) and not isinstance(e[1], int):
                    raise TypeError('The ranges need to be given either as float or as int!')
                self.parameter_range.append([e[0], e[1]])
            else:
                raise TypeError('the ranges have to be provided either as list of tuples or list of lists')
        if not (isinstance(resolution, tuple) or isinstance(resolution, list)):
            raise TypeError('The resolutions have to be provided as list or tuples')
        if len(resolution) != len(self.parameter_range):
            raise ValueError('Parameter ranges and resolutions are of inconsistent lengths '
                             + f'({len(self.parameter_range)} <> {len(resolution)})')
        for e in resolution:
            if not isinstance(e, float) and not isinstance(e, int):
                raise TypeError('The resolutions need to be given either as float or as int!')
            self.resolution.append(e)
        if not isinstance(fitness_func, Callable):
            raise TypeError('The fitness function has to be callable!')
        self.fitness_func = fitness_func
        self.pre_process = pre_process
        self.post_process = post_process

        self.parallelism = parallelism
        if isinstance(self.parallelism, bool) and self.parallelism:
            self.parallelism = cpu_count()
        elif self.parallelism is None:
            self.parallelism = 1
        elif isinstance(self.parallelism, int) and self.parallelism < 1:
            raise ValueError('the value for parallelism needs to be either None, True or > 0')

        # pre-computation
        self.num_parameters = len(parameter_range)
        self.extent = []
        for i in range(self.num_parameters):
            self.extent.append(int((self.parameter_range[i][1] - self.parameter_range[i][0]) / self.resolution[i]))

        self.hsum = [1] * self.num_parameters
        if self.num_parameters > 1:
            for i in range(self.num_parameters-2, -1, -1):
                self.hsum[i] = self.hsum[i+1]*self.extent[i+1]
        self.num_samples = self.hsum[0] * self.extent[0]
        self.parallelism = self.parallelism if self.parallelism < self.num_samples else self.num_samples

        self.p_range = list(range(0, self.num_samples, int(self.num_samples/self.parallelism)))
        self.p_range.append(self.num_samples)
        if self.parallelism > 1:
            print(f'Generating worker pool with {self.parallelism} workers')
        self.pool = Pool(self.parallelism)

        if self.save_all_results:
            self.all_results = []

    def run(self) -> None:
        r"""
        Conducts the parameter sweep

        Notes
        -----
        Here, all values within the specified range and resolution will be evaluation according to their fitness.
        For distribution of the parameters between workers, a each parameter set is mapped to a unique value within
        a monotonous series and the solution evaluated. The first worker provides a visual indicator for the progress
        of the sweeps
        """
        def inner_loop(n_worker: int):
            # defining the loop that should be executed by the worker
            best_solution_l = [-1.] * self.num_parameters
            best_solution_fitness_l = 0.
            num_samples_l = self.p_range[n_worker+1] - self.p_range[n_worker]  # get the range of values for this specific worker
            if self.save_all_results:
                all_results_l = []
            # only show progress bar for process 0
            disable_tqdm = n_worker != 0
            if not disable_tqdm and self.parallelism > 1:
                print(f'The progress bar only provides an indicator of the progress based on process {n_worker} '
                      + f'with {num_samples_l} of {self.num_samples} values')

            # To provide better feedback via the progress bar, the data range is mapped to a 1D space
            # and from there the information retrieved
            for i in tqdm(range(num_samples_l), disable=disable_tqdm):
                i_total = i + self.p_range[n_worker]
                params = self._mapping_1D_to_Range(i_total)
                for j in range(self.num_parameters):
                    params[j] = params[j] * self.resolution[j] + self.parameter_range[j][0]
                r_l = self.fitness_func(self, params)
                if isinstance(r_l, tuple):
                    fit_l = r_l[0]
                    if self.save_all_results:
                        all_results_l.append((params, fit_l, r_l[1:]))
                else:
                    fit_l = r_l
                    if self.save_all_results:
                        all_results_l.append((params, fit_l))

                if fit_l > best_solution_fitness_l:
                    best_solution_l = params
                    best_solution_fitness_l = fit_l
            if self.save_all_results:
                return (best_solution_l, best_solution_fitness_l, all_results_l)
            else:
                return (best_solution_l, best_solution_fitness_l)

        if self.pre_process:
            self.pre_process(self)

        # actually run the process
        tic = time.perf_counter()
        result = self.pool.map(inner_loop, range(self.parallelism))
        toc = time.perf_counter()
        self.elapsed_time = toc - tic

        # reduce the result array
        for r in result:
            if r[1] > self.best_fitness:
                self.best_fit = r[0]
                self.best_fitness = r[1]
            if self.save_all_results:
                self.all_results += r[2]

        if self.post_process:
            self.post_process(self)

    def BestFit(self) -> tuple:
        r"""
        Returns the best fit and associated best fitness

        Returns
        -------
        Tuple of (best fit, best fitness)
        """
        return self.best_fit, self.best_fitness

    def ElapsedTime(self) -> float:
        r"""
        provides elapsed time of the sweep

        Returns
        -------
        Time in s as float
        """
        return self.elapsed_time

    def AllResults(self) -> list:
        if self.save_all_results:
            return self.all_results
        else:
            raise ValueError('the option for saving all results was deactivated, cannot provide anything')

    def _mapping_Range_to_1D(self, params) -> int:
        r"""
        maps a range to a unique integer value

        Parameters
        ----------
        params: list
            list of parameters to be mapped to a unique value

        Returns
        -------
        unique integer value
        """
        if len(params) != self.num_parameters:
            raise ValueError('the provided parameter are not consistent')
        i = 0
        for n in range(params):
            i += params[n] * self.hsum[n]
        return i

    def _mapping_1D_to_Range(self, i: int):
        r"""
        maps a unique value to the associated parameter set

        Parameters
        ----------
        i: int
            unique integer value

        Returns
        -------
        list of parameters
        """
        iloc = i
        params = [0] * self.num_parameters
        for n in range(self.num_parameters):
            params[n] = int(iloc / self.hsum[n])
            iloc -= params[n] * self.hsum[n]
        return params
