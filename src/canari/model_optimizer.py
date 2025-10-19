"""
This module automates the search for optimal hyperparameters of a
:class:`~canari.model.Model` instance by leveraging external libraries.
"""

import platform
from typing import Callable, Dict, Optional
import numpy as np

import signal
from ray import tune
from ray.tune import Callback
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from canari import Model

signal.signal(signal.SIGSEGV, lambda signum, frame: None)

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


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
        mode: Optional[str] = "min",
        algorithm: Optional[str] = "default",
    ):
        """
        Initialize the ModelOptimizer.
        """

        self.model = model
        self.model_optim = None
        self.param_optim = None
        self._param_space = param_space
        self._train_data = train_data
        self._validation_data = validation_data
        self._num_optimization_trial = num_optimization_trial
        self._grid_search = grid_search
        self._mode = mode
        self._trial_count = 0
        self._algorithm = algorithm
        self._backend = "optuna" if platform.system() == "Windows" else "ray"

    def objective(self, config: Dict) -> Dict:
        """
        Returns a metric that is used for optimization

        Returns:
            dict: Metric used for optimization.
        """
        trained_model, *_ = self.model(config, self._train_data, self._validation_data)
        _metric = trained_model.metric_optim

        metric = {}
        metric["metric"] = _metric
        return metric

    def optimize(self):
        """
        Run optimziation
        """

        if self._backend == "ray":
            self._ray_optimizer()
        elif self._backend == "optuna":
            self._optuna_optimizer()

    def get_best_model(self) -> Model:
        """
        Retrieve the optimized model instance after running optimization.

        Returns:
            :class:`~canari.model.Model`: Model instance initialized with the best
                                            hyperparameter values.

        """
        return self.model_optim

    def get_best_param(self) -> Dict:
        """
        Retrieve the optimized parameters after running optimization.

        Returns:
            dict: Best hyperparameter values.

        """
        return self.param_optim

    def _ray_optimizer(self):
        """
        Run hyperparameter optimization over the defined search space.
        """
        search_config = self._ray_build_search_space()

        if self._grid_search:
            total_trials = 1
            for v in self._param_space.values():
                total_trials *= len(v)
            custom_logger = self._ray_progress_callback(total_samples=total_trials)

            optimizer_runner = tune.run(
                self.objective,
                config=search_config,
                name="Model_optimizer",
                num_samples=1,
                verbose=0,
                raise_on_failed_trial=False,
                callbacks=[custom_logger],
            )
        else:
            custom_logger = self._ray_progress_callback(
                total_samples=self._num_optimization_trial
            )
            if self._algorithm == "default":
                optimizer_runner = tune.run(
                    self.objective,
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
                    self.objective,
                    config=search_config,
                    name="Model_optimizer",
                    num_samples=self._num_optimization_trial,
                    scheduler=scheduler,
                    verbose=0,
                    raise_on_failed_trial=False,
                    callbacks=[custom_logger],
                )
            else:
                raise ValueError(
                    "algorithm must be 'default' (OptunaSearch) or 'parallel' (ASHAScheduler)"
                )

        # Best params & model
        self.param_optim = optimizer_runner.get_best_config(
            metric="metric", mode=self._mode
        )
        best_trial = optimizer_runner.get_best_trial(metric="metric", mode=self._mode)
        best_sample_number = custom_logger.trial_sample_map.get(
            best_trial.trial_id, "Unknown"
        )

        best_model, *_ = self.model(
            self.param_optim, self._train_data, self._validation_data
        )
        self.model_optim = best_model

        print("-----")
        print(f"Optimal parameters at trial #{best_sample_number}: {self.param_optim}")
        print("-----")

    def _ray_build_search_space(self) -> Dict:
        """
        Convert param_space to Ray Tune search space objects.
        """
        search_config = {}
        for param_name, values in self._param_space.items():
            # Grid search
            if self._grid_search:
                search_config[param_name] = tune.grid_search(values)
                continue

            # Random search
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

        return search_config

    def _ray_progress_callback(self, total_samples: int) -> Callback:
        """Create a Ray Tune callback bound to this optimizer instance."""

        class _Progress(Callback):
            def __init__(self, total):
                self.total_samples = total
                self.current_sample = 0
                self.trial_sample_map = {}

            def on_trial_result(self, iteration, trial, result, **info):
                self.current_sample += 1
                metric = result["metric"]
                params = trial.config

                self.trial_sample_map[trial.trial_id] = self.current_sample

                width = len(f"{self.total_samples}/{self.total_samples}")
                sample_str = f"{self.current_sample}/{self.total_samples}".rjust(width)
                print(f"# {sample_str} - Metric: {metric:.3f} - Parameter: {params}")

        return _Progress(total_samples)

    def _optuna_optimizer(self):
        """
        Optuna optimizer
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
            self._optuna_objective,
            n_trials=self._num_optimization_trial,
            callbacks=[self._optuna_log_trial],
        )

        self.param_optim = study.best_params
        self.model_optim, *_ = self.model(
            self.param_optim, self._train_data, self._validation_data
        )

        print("-----")
        print(
            f"Optimal parameters at trial #{study.best_trial.number + 1}: "
            f"{self.param_optim}"
        )
        print(f"Best metric value: {study.best_value:.4f}")
        print("-----")

    def _optuna_objective(self, trial: optuna.Trial):
        """
        Objective function
        """

        param = self._optuna_build_search_space(trial)
        metric = self.objective(param)
        return metric["metric"]

    def _optuna_build_search_space(self, trial: optuna.Trial) -> Dict:
        """
        Build parameter suggestions for Optuna from self._param_space.

        Args:
            trial (optuna.Trial): Optuna trial object used to sample parameters.

        Returns:
            Dict[str, float | int]: Dictionary of parameter names mapped to suggested values.
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
        return param

    def _optuna_log_trial(self, study: optuna.Study, trial: optuna.Trial):
        """
        Custom logging of trial progress.
        """

        self._trial_count += 1
        trial_id = f"{self._trial_count}/{self._num_optimization_trial}".rjust(
            len(f"{self._num_optimization_trial}/{self._num_optimization_trial}")
        )
        print(f"# {trial_id} - Metric: {trial.value:.4f} - Parameter: {trial.params}")

        if trial.number == study.best_trial.number:
            print(
                f" -> New best trial #{trial.number + 1} with metric: {trial.value:.4f}"
            )
