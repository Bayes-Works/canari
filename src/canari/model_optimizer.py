"""
This module automates the search for optimal hyperparameters of a
:class:`~canari.model.Model` instance by leveraging the Ray Tune
external library.
"""

import signal
from typing import Callable, Optional
from ray import tune
from ray.tune import Callback
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from canari.data_process import DataProcess

# Ignore segmentation fault signals
signal.signal(signal.SIGSEGV, lambda signum, frame: None)


class ModelOptimizer:
    """
    Optimize hyperparameters for :class:`~canari.model.Model` using the Ray Tune
    external library.

    Args:
        initialize_model (Callable):
            Function that returns a model instance given a model configuration.
        train (Callable):
            Function that define the training procedure for a model, it should return
            `trained_model.early_stop_metric`.
        param_space (Dict[str, list]):
            Parameter search space: two-value lists [min, max] for defining the
            bounds of the optimization.
        data_processor (DataProcess):
            DataProcess instance for data preparation.
        num_optimization_trial (int, optional):
            Number of random search trials (ignored for grid search). Defaults to 50.
        grid_search (bool, optional):
            If True, perform grid search. Defaults to False.
        algorithm (str, optional):
            Search algorithm: 'default' (OptunaSearch) or 'parallel' (ASHAScheduler).
            Defaults to 'OptunaSearch'.

    Attributes:
        model_optim :
            The best model instance initialized with optimal parameters after running optimize().
        param_optim (Dict[str, Any]):
            The best hyperparameter configuration found during optimization.
    """

    def __init__(
        self,
        initialize_model: Callable,
        train: Callable,
        param_space: dict,
        data_processor: DataProcess,
        num_optimization_trial: Optional[int] = 50,
        grid_search: Optional[bool] = False,
        algorithm: Optional[str] = "default",
    ):
        """
        Initialize the ModelOptimizer.
        """

        self._initialize_model = initialize_model
        self._train = train
        self._param_space = param_space
        self._data_processor = data_processor
        self._num_optimization_trial = num_optimization_trial
        self._grid_search = grid_search
        self._algorithm = algorithm
        self.model_optim = None
        self.param_optim = None

    def optimize(self):
        """
        Run hyperparameter optimization over the defined search space.
        """

        # Function for optimization
        def objective(
            config,
        ):
            model = self._initialize_model(config)
            trained_model, *_ = self._train(
                model,
                self._data_processor,
            )
            tune.report({"metric": trained_model.early_stop_metric})

        # Parameter space
        search_config = {}
        if self._grid_search:
            total_trials = 1
            for param_name, values in self._param_space.items():
                search_config[param_name] = tune.grid_search(values)
                total_trials *= len(values)

            custom_logger = _CustomLogger(total_samples=total_trials)
            optimizer_runner = tune.run(
                objective,
                config=search_config,
                name="Model_optimizer",
                num_samples=1,
                verbose=0,
                raise_on_failed_trial=False,
                callbacks=[custom_logger],
            )
        else:
            for param_name, values in self._param_space.items():
                if isinstance(values, list) and len(values) == 2:
                    low, high = values
                    if isinstance(low, int) and isinstance(high, int):
                        search_config[param_name] = tune.randint(low, high)
                    elif isinstance(low, float) and isinstance(high, float):
                        search_config[param_name] = tune.uniform(low, high)
                    else:
                        raise ValueError(
                            f"Unsupported type for parameter {param_name}: {values}"
                        )
                else:
                    raise ValueError(
                        f"Parameter {param_name} should be a list of two values (min, max)."
                    )

            # Run optimization
            custom_logger = _CustomLogger(total_samples=self._num_optimization_trial)
            if self._algorithm == "default":
                optimizer_runner = tune.run(
                    objective,
                    config=search_config,
                    search_alg=OptunaSearch(metric="metric", mode="min"),
                    name="Model_optimizer",
                    num_samples=self._num_optimization_trial,
                    verbose=0,
                    raise_on_failed_trial=False,
                    callbacks=[custom_logger],
                )
            elif self._algorithm == "parallel":
                scheduler = ASHAScheduler(metric="metric", mode="min")
                optimizer_runner = tune.run(
                    objective,
                    config=search_config,
                    name="Model_optimizer",
                    num_samples=self._num_optimization_trial,
                    scheduler=scheduler,
                    verbose=0,
                    raise_on_failed_trial=False,
                    callbacks=[custom_logger],
                )

        # Get the optimal parameters
        self.param_optim = optimizer_runner.get_best_config(metric="metric", mode="min")
        best_trial = optimizer_runner.get_best_trial(metric="metric", mode="min")
        best_sample_number = custom_logger.trial_sample_map.get(
            best_trial.trial_id, "Unknown"
        )

        # Get the optimal model
        self.model_optim = self._initialize_model(self.param_optim)

        # Print optimal parameters
        print("-----")
        print(f"Optimal parameters at trial #{best_sample_number}: {self.param_optim}")
        print("-----")

    def get_best_model(self):
        """
        Retrieve the optimized model instance after running optimization.

        Returns:
            Any: Model instance initialized with the best hyperparameter values.

        """
        return self.model_optim


class _CustomLogger(Callback):
    """
    Ray Tune callback for custom logging of trial progress.

    Attributes:
        total_samples (int): Total number of expected trials.
        current_sample (int): Counter of completed samples.
        trial_sample_map (Dict[str, int]):
            Maps trial IDs to their corresponding sample index.
    """

    def __init__(self, total_samples):
        """
        Initialize the _CustomLogger.

        Args:
            total_samples (int): Total number of optimization trials.
        """

        self.total_samples = total_samples
        self.current_sample = 0
        self.trial_sample_map = {}

    def on_trial_result(self, iteration, trial, result, **info):
        """
        Log progress when a trial reports results.

        Increments the sample counter, records a mapping from trial ID to
        the sample index, and prints a formatted line containing the running
        sample count, reported metric, and trial parameters.

        Args:
            iteration (int): Current iteration number of Ray Tune.
            trial (Trial): The Ray Tune Trial object.
            result (Dict[str, Any]): Dictionary of trial results; must include key 'metric'.
            **info: Additional callback info.
        """

        self.current_sample += 1
        params = trial.config
        metric = result["metric"]

        # Store sample number mapped to the trial ID
        self.trial_sample_map[trial.trial_id] = self.current_sample

        # Ensure sample count formatting consistency
        sample_str = f"{self.current_sample}/{self.total_samples}".rjust(
            len(f"{self.total_samples}/{self.total_samples}")
        )

        print(f"# {sample_str} - Metric: {metric:.3f} - Parameter: {params}")
