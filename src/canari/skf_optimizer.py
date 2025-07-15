"""
This module automates the search for optimal hyperparameters of a
:class:`~canari.skf.SKF` instance by leveraging the Optuna
external library.
"""

from typing import Callable, Dict, Optional

import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


class SKFOptimizer:
    """
    Optimize hyperparameters for :class:`~canari.skf.SKF` using the Optuna external library.

    Args:
        initialize_skf (Callable): Function that returns an SKF instance given a set of parameter.
        model_param (dict): Serializable dictionary for :class:`~canari.model.Model` obtained from
                            :meth:`~canari.model.Model.get_dict`.
        param_space (dict): Parameter search space: two-value lists [min, max] for defining the
                            bounds of the optimization.
        data (dict): Input data for adding synthetic anomalies.
        detection_threshold (float, optional): Threshold for the target maximal anomaly detection rate.
                                                Defaults to 0.5.
        false_rate_threshold (float, optional): Threshold for the maximal false detection rate.
                                                Defaults to 0.0.
        max_timestep_to_detect (int, optional): Maximum number of timesteps to allow detection.
                                                Defaults to None (to the end of time series).
        num_synthetic_anomaly (int, optional): Number of synthetic anomalies to add. This will create as
                            many time series, because one time series contains only one
                            anomaly. Defaults to 50.
        num_optimization_trial (int, optional): Number of trials for optimizer. Defaults to 50.
        grid_search (bool, optional): If True, perform grid search. Defaults to False.

    Attributes:
        detection_threshold: Threshold for detection rate for anomaly detection.
        false_rate_threshold: Threshold for false rate.
    """

    def __init__(
        self,
        initialize_skf: Callable,
        model_param: dict,
        param_space: Dict[str, list],
        data: dict,
        detection_threshold: Optional[float] = 0.5,
        false_rate_threshold: Optional[float] = 0.0,
        max_timestep_to_detect: Optional[int] = None,
        num_synthetic_anomaly: Optional[int] = 50,
        num_optimization_trial: Optional[int] = 50,
        grid_search: Optional[bool] = False,
    ):
        """
        Initializes the SKFOptimizer.
        """

        self._initialize_skf = initialize_skf
        self._model_param = model_param
        self._param_space = param_space
        self._data = data
        self.detection_threshold = detection_threshold
        self.false_rate_threshold = false_rate_threshold
        self._max_timestep_to_detect = max_timestep_to_detect
        self._num_synthetic_anomaly = num_synthetic_anomaly
        self._num_optimization_trial = num_optimization_trial
        self._grid_search = grid_search
        self.skf_optim = None
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

        detection_rate = trial.user_attrs["detection_rate"]
        false_rate = trial.user_attrs["false_rate"]
        false_alarm = trial.user_attrs["false_alarm"]

        print(
            f"# {trial_id} - Metric: {trial.value:.5f} - Detection rate: {detection_rate:.2f} - "
            f"False rate: {false_rate:.2f} - False alarm in training data: {false_alarm} - Param: {trial.params}"
        )

        if trial.number == study.best_trial.number:
            print(
                f" -> New best trial #{trial.number + 1} with metric: {trial.value:.5f}"
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
                    log_uniform = low > 0 and high > 0
                    param[name] = trial.suggest_float(name, low, high, log=log_uniform)

        skf = self._initialize_skf(param, self._model_param)
        slope = param.get("slope")

        detection_rate, false_rate, false_alarm = skf.detect_synthetic_anomaly(
            data=self._data,
            num_anomaly=self._num_synthetic_anomaly,
            slope_anomaly=slope,
            max_timestep_to_detect=self._max_timestep_to_detect,
        )

        if (
            detection_rate < self.detection_threshold
            or false_rate > self.false_rate_threshold
            or false_alarm == "Yes"
        ):
            metric = np.abs(self._param_space["slope"][1])  # upper bound of slope
        else:
            metric = np.abs(slope)

        # Save extra info for callback
        trial.set_user_attr("detection_rate", detection_rate)
        trial.set_user_attr("false_rate", false_rate)
        trial.set_user_attr("false_alarm", false_alarm)

        return metric

    def optimize(self):
        """
        Run hyperparameter optimization over the defined search space.
        """

        if self._grid_search:
            sampler = optuna.samplers.GridSampler(self._param_space)
            self._num_optimization_trial = int(
                np.prod([len(v) for v in self._param_space.values()])
            )
        else:
            sampler = optuna.samplers.TPESampler()
            self._num_optimization_trial = self._num_optimization_trial

        print("-----")
        print("SKF optimization starts")
        study = optuna.create_study(direction="minimize", sampler=sampler)

        study.optimize(
            self._objective,
            n_trials=self._num_optimization_trial,
            callbacks=[self._log_trial],
        )

        self.param_optim = study.best_params
        self.skf_optim = self._initialize_skf(self.param_optim, self._model_param)

        print("-----")
        print(
            f"Optimal parameters at trial #{study.best_trial.number + 1}: "
            f"{self.param_optim}"
        )
        print(f"Best metric value: {study.best_value:.5f}")
        print("-----")

    def get_best_model(self):
        """
        Retrieves the SKF instance initialized with the best parameters.

        Returns:
            Any: SKF instance corresponding to the optimal configuration.
        """
        return self.skf_optim

    def get_best_param(self):
        """
        Retrieve the optimized parameters after running optimization.

        Returns:
            dict: Best hyperparameter values.

        """
        return self.param_optim
