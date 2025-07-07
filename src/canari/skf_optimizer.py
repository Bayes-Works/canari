import signal
import logging
import itertools
from typing import Callable, Optional

import optuna
import numpy as np

# Suppress segmentation faults (optional)
signal.signal(signal.SIGSEGV, lambda signum, frame: None)

# Silence Optuna's default logger
optuna.logging.set_verbosity(optuna.logging.WARNING)


class SKFOptimizer:
    """
    Optimize hyperparameters for :class:`~canari.skf.SKF` using Optuna.

    Args:
        initialize_skf (Callable): Function that returns an SKF instance given a configuration.
        model_param (dict): Serializable dict from `Model.get_dict()`.
        param_space (dict): For grid search, pass list of values; for random search, use [min, max].
        data (dict): Input data for injecting synthetic anomalies.
        detection_threshold (float): Minimum required detection rate. Defaults to 0.5.
        false_rate_threshold (float): Maximum allowed false detection rate. Defaults to 0.0.
        max_timestep_to_detect (int, optional): Max timesteps to detect. Defaults to full series.
        num_synthetic_anomaly (int): How many synthetic anomalies to inject. Defaults to 50.
        num_optimization_trial (int): Number of trials (ignored for grid search). Defaults to 50.
        grid_search (bool): Whether to perform grid search. Defaults to False.
        algorithm (str): Ignored in Optuna version.
    """

    def __init__(
        self,
        initialize_skf: Callable,
        model_param: dict,
        param_space: dict,
        data: dict,
        detection_threshold: Optional[float] = 0.5,
        false_rate_threshold: Optional[float] = 0.0,
        max_timestep_to_detect: Optional[int] = None,
        num_synthetic_anomaly: Optional[int] = 50,
        num_optimization_trial: Optional[int] = 50,
        grid_search: Optional[bool] = False,
        algorithm: Optional[str] = "default",
    ):
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
        self._algorithm = algorithm

        self.skf_optim = None
        self.param_optim = None
        self._trial_count = 0
        self._total_trials = 0

    def _optuna_objective(self, trial: optuna.Trial):
        if self._grid_search:
            config = trial.params
        else:
            config = {}
            for name, bounds in self._param_space.items():
                if isinstance(bounds, list) and len(bounds) == 2:
                    low, high = bounds
                    if isinstance(low, int) and isinstance(high, int):
                        config[name] = trial.suggest_int(name, low, high)
                    elif isinstance(low, float) and isinstance(high, float):
                        if low < 0 or high < 0:
                            config[name] = trial.suggest_float(name, low, high)
                        else:
                            config[name] = trial.suggest_float(
                                name, low, high, log=True
                            )
                    else:
                        raise ValueError(f"Invalid param type for {name}: {bounds}")
                else:
                    raise ValueError(f"{name} must be [min, max] for random search")

        skf = self._initialize_skf(config, self._model_param)
        slope = config.get("slope", 1.0)

        detection_rate, false_rate, false_alarm_train = skf.detect_synthetic_anomaly(
            data=self._data,
            num_anomaly=self._num_synthetic_anomaly,
            slope_anomaly=slope,
            max_timestep_to_detect=self._max_timestep_to_detect,
        )

        if (
            detection_rate < self.detection_threshold
            or false_rate > self.false_rate_threshold
            or false_alarm_train == "Yes"
        ):
            metric = 2 + 5 * slope
        else:
            metric = detection_rate + 5 * abs(slope)

        # Custom logging
        self._trial_count += 1
        trial_id = f"{self._trial_count}/{self._total_trials}".rjust(
            len(f"{self._total_trials}/{self._total_trials}")
        )
        print(
            f"# {trial_id} - Metric: {metric:.3f} - Detection rate: {detection_rate:.2f} - "
            f"False rate: {false_rate:.2f} - False alarm in train: {false_alarm_train} - "
            f"Parameter: {config}"
        )

        return metric

    def optimize(self):
        """Run the optimization process using Optuna."""
        study = optuna.create_study(direction="minimize")

        if self._grid_search:
            keys = list(self._param_space.keys())
            values = [self._param_space[k] for k in keys]
            combos = list(itertools.product(*values))
            self._total_trials = len(combos)

            print(f"Running grid search over {self._total_trials} combinations...")

            for combo in combos:
                study.enqueue_trial(dict(zip(keys, combo)))

            study.optimize(
                self._optuna_objective,
                n_trials=self._total_trials,
            )
        else:
            self._total_trials = self._num_optimization_trial
            print(f"Running random search over {self._total_trials} trials...")

            study.optimize(
                self._optuna_objective,
                n_trials=self._num_optimization_trial,
            )

        self.param_optim = study.best_params
        self.skf_optim = self._initialize_skf(self.param_optim, self._model_param)

        print("-----")
        print(
            f"Optimal parameters at trial #{study.best_trial.number}: {self.param_optim}"
        )
        print(f"Best metric value: {study.best_value:.4f}")
        print("-----")

    def get_best_model(self):
        """Return the optimized SKF instance."""
        return self.skf_optim

    def get_best_param(self):
        """Return the best hyperparameter configuration found."""
        return self.param_optim
