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
        print(f"# {trial_id} - Metric: {trial.value:.4f} - Parameter: {trial.params}")

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
