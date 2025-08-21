"""
This module automates the search for optimal hyperparameters of a
:class:`~canari.model.Model` instance by leveraging the Optuna
external library.
"""

import platform
from typing import Callable, Dict, Optional
import numpy as np

if platform.system() == "Windows":
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    class ModelOptimizer:
        """
        Optimize hyperparameters for :class:`~canari.model.Model` using Optuna based on
        the metric saved in :attr:`~canari.model.Model.metric_optim`.

        Args:
            model (Callable): Function that returns a model instance given a set of parameter.
            param_space (Dict[str, list]): For random search, use [low, high]. For grid search, use full list of values.
            train_data (Dict[str, np.ndarray], optional): Training data.
            validation_data (Dict[str, np.ndarray], optional): Validation data.
            num_optimization_trial (int, optional): Number of optimization trials (ignored in grid search).
            grid_search (bool, optional): Whether to perform grid search. Defaults to False.
            mode (str, optional): Direction for optimization: 'min' (default) or 'max'.
        """

        def __init__(
            self,
            model: Callable,
            param_space: Dict[str, list],
            train_data: Optional[dict] = None,
            validation_data: Optional[dict] = None,
            num_optimization_trial: int = 50,
            grid_search: bool = False,
            mode: str = "min",
        ):
            """
            Initialize the ModelOptimizer.
            """

            self.model_objective = model
            self._param_space = param_space
            self._train_data = train_data
            self._validation_data = validation_data
            self._num_optimization_trial = num_optimization_trial
            self._grid_search = grid_search
            self._mode = mode
            self.model_optim = None
            self.param_optim = None
            self._trial_count = 0

        def _log_trial(self, study: optuna.Study, trial: optuna.Trial):
            """
            Custom logging of trial progress.
            """

            self._trial_count += 1
            trial_id = f"{self._trial_count}/{self._num_optimization_trial}".rjust(
                len(f"{self._num_optimization_trial}/{self._num_optimization_trial}")
            )
            print(
                f"# {trial_id} - Metric: {trial.value:.4f} - Parameter: {trial.params}"
            )

            if trial.number == study.best_trial.number:
                print(
                    f" -> New best trial #{trial.number + 1} with metric: {trial.value:.4f}"
                )

        def _objective(self, trial: optuna.Trial):
            """
            Objective function
            """

            param = {}
            if self._grid_search:
                for name, values in self._param_space.items():
                    param[name] = trial.suggest_categorical(name, values)
            else:
                for name, bounds in self._param_space.items():
                    low, high = bounds
                    if all(isinstance(x, int) for x in bounds):
                        param[name] = trial.suggest_int(name, low, high)
                    else:
                        param[name] = trial.suggest_float(name, low, high)

            trained_model, *_ = self.model_objective(
                param, self._train_data, self._validation_data
            )
            return trained_model.metric_optim

        def optimize(self):
            """
            Run hyperparameter optimization over the defined search space.
            """

            direction = "minimize" if self._mode == "min" else "maximize"

            if self._grid_search:
                sampler = optuna.samplers.GridSampler(self._param_space)
                self._num_optimization_trial = int(
                    np.prod([len(v) for v in self._param_space.values()])
                )
            else:
                sampler = optuna.samplers.TPESampler()

            print("-----")
            print("Model optimization starts")
            study = optuna.create_study(direction=direction, sampler=sampler)

            study.optimize(
                self._objective,
                n_trials=self._num_optimization_trial,
                callbacks=[self._log_trial],
            )

            self.param_optim = study.best_params
            self.model_optim, *_ = self.model_objective(
                self.param_optim, self._train_data, self._validation_data
            )

            print("-----")
            print(
                f"Optimal parameters at trial #{study.best_trial.number + 1}: "
                f"{self.param_optim}"
            )
            print(f"Best metric value: {study.best_value:.4f}")
            print("-----")

        def get_best_model(self):
            """
            Retrieve the optimized model instance after running optimization.

            Returns:
                :class:`~canari.model.Model`:: Model instance initialized with the best
                                                hyperparameter values.

            """
            return self.model_optim

        def get_best_param(self):
            """
            Retrieve the optimized parameters after running optimization.

            Returns:
                dict: Best hyperparameter values.

            """
            return self.param_optim

else:
    import signal
    from ray import tune
    from ray.tune import Callback
    from ray.tune.search.optuna import OptunaSearch
    from ray.tune.schedulers import ASHAScheduler

    # Ignore segmentation fault signals
    signal.signal(signal.SIGSEGV, lambda signum, frame: None)

    class ModelOptimizer:
        """
        Optimize hyperparameters for :class:`~canari.model.Model` using the Ray Tune
        external library using the metric :attr:`~canari.model.Model.metric_optim`.

        Args:
            model (Callable):
                Function that returns a model instance given a model configuration.
            param_space (Dict[str, list]):
                Parameter search space: two-value lists [min, max] for defining the
                bounds of the optimization.
            train_data (Dict[str, np.ndarray], optional):
                Training data.
            validation_data (Dict[str, np.ndarray], optional):
                Validation data.
            num_optimization_trial (int, optional):
                Number of random search trials (ignored for grid search). Defaults to 50.
            grid_search (bool, optional):
                If True, perform grid search. Defaults to False.
            algorithm (str, optional):
                Search algorithm: 'default' (OptunaSearch) or 'parallel' (ASHAScheduler).
                Defaults to 'OptunaSearch'.
            mode (str, optional): Direction for optimization stopping: 'min' (default).

        Attributes:
            model_optim :
                The best model instance initialized with optimal parameters after running optimize().
            param_optim (Dict):
                The best hyperparameter configuration found during optimization.
        """

        def __init__(
            self,
            model: Callable,
            param_space: dict,
            train_data: Optional[dict] = None,
            validation_data: Optional[dict] = None,
            num_optimization_trial: Optional[int] = 50,
            grid_search: Optional[bool] = False,
            algorithm: Optional[str] = "default",
            mode: Optional[str] = "min",
        ):
            """
            Initialize the ModelOptimizer.
            """

            self.model_objective = model
            self._param_space = param_space
            self._train_data = train_data
            self._validation_data = validation_data
            self._num_optimization_trial = num_optimization_trial
            self._grid_search = grid_search
            self._algorithm = algorithm
            self._mode = mode
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
                trained_model, *_ = self.model_objective(
                    config, self._train_data, self._validation_data
                )
                tune.report({"metric": trained_model.metric_optim})

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
                custom_logger = _CustomLogger(
                    total_samples=self._num_optimization_trial
                )
                if self._algorithm == "default":
                    optimizer_runner = tune.run(
                        objective,
                        config=search_config,
                        search_alg=OptunaSearch(metric="metric", mode=self._mode),
                        name="Model_optimizer",
                        num_samples=self._num_optimization_trial,
                        verbose=0,
                        raise_on_failed_trial=False,
                        callbacks=[custom_logger],
                    )
                elif self._algorithm == "parallel":
                    scheduler = ASHAScheduler(metric="metric", mode=self._mode)
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
            self.param_optim = optimizer_runner.get_best_config(
                metric="metric", mode=self._mode
            )
            best_trial = optimizer_runner.get_best_trial(
                metric="metric", mode=self._mode
            )
            best_sample_number = custom_logger.trial_sample_map.get(
                best_trial.trial_id, "Unknown"
            )

            # Get the optimal model
            self.model_optim = self.model_objective(
                self.param_optim, self._train_data, self._validation_data
            )

            # Print optimal parameters
            print("-----")
            print(
                f"Optimal parameters at trial #{best_sample_number}: {self.param_optim}"
            )
            print("-----")

        def get_best_model(self):
            """
            Retrieve the optimized model instance after running optimization.

            Returns:
                :class:`~canari.model.Model`:: Model instance initialized with the best
                                                hyperparameter values.

            """
            return self.model_optim

        def get_best_param(self):
            """
            Retrieve the optimized parameters after running optimization.

            Returns:
                dict: Best hyperparameter values.

            """
            return self.param_optim

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
