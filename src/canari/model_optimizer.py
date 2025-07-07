"""
This module automates the search for optimal hyperparameters of a
:class:`~canari.model.Model` instance by leveraging the Optuna
external library.
"""

from typing import Callable, Dict, Optional
import optuna
import numpy as np

optuna.logging.set_verbosity(optuna.logging.WARNING)


class ModelOptimizer:
    """
    Optimize hyperparameters for :class:`~canari.model.Model` using Optuna.

    Args:
        model (Callable): Function that returns a model instance given a config.
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

    def _objective(self, trial: optuna.Trial):
        """
        Objective function
        """
        config = {}

        if self._grid_search:  # GridSampler – use suggest_categorical
            for name, values in self._param_space.items():
                config[name] = trial.suggest_categorical(name, values)
        else:  # Random/TPE – use bounds
            for name, bounds in self._param_space.items():
                low, high = bounds
                if all(isinstance(x, int) for x in bounds):
                    config[name] = trial.suggest_int(name, low, high)
                else:
                    config[name] = trial.suggest_float(name, low, high)

        trained_model, *_ = self.model_objective(
            config, self._train_data, self._validation_data
        )
        return trained_model.metric_optim

    def _log_trial_result(self, study: optuna.Study, trial: optuna.Trial):
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
                f" -> New best trial #{trial.number + 1} with value: {trial.value:.4f}"
            )

    # ------------------------------------------------------------------ #
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
            sampler = optuna.samplers.TPESampler(seed=42)

        print("Optimization starts")
        study = optuna.create_study(direction=direction, sampler=sampler)

        self._trial_count = 0
        study.optimize(
            self._objective,
            n_trials=self._num_optimization_trial,
            callbacks=[self._log_trial_result],
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
