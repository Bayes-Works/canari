"""
This module automates the search for optimal hyperparameters of a
:class:`~canari.skf.SKF` instance by leveraging the Optuna
external library.
"""

import platform
from typing import Callable, Dict, Optional
import numpy as np

import signal
from ray import tune
from ray.tune import Callback
from typing import Callable, Optional
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from canari import SKF

signal.signal(signal.SIGSEGV, lambda signum, frame: None)

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


class SKFOptimizer:
    """
    Optimize hyperparameters for :class:`~canari.skf.SKF` using the Ray Tune external library.

    Args:
        skf (Callable): Function that returns an SKF instance given a configuration.
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
        num_optimization_trial (int, optional): Number of trials for optimizer. Defaults to 50.
        grid_search (bool, optional): If True, perform grid search. Defaults to False.
        algorithm (str, optional): Search algorithm: 'default' (OptunaSearch) or 'parallel' (ASHAScheduler). Defaults to 'OptunaSearch'.

    Attributes:
        skf_optim: Best SKF instance after optimization.
        param_optim (dict): Best hyperparameter configuration.
        detection_threshold: Threshold for detection rate for anomaly detection.
        false_rate_threshold: Threshold for false rate.
    """

    def __init__(
        self,
        skf: Callable,
        model_param: dict,
        param_space: dict,
        data: dict,
        detection_threshold: Optional[float] = 0.5,
        false_rate_threshold: Optional[float] = 0.0,
        max_timestep_to_detect: Optional[int] = None,
        num_optimization_trial: Optional[int] = 50,
        grid_search: Optional[bool] = False,
        algorithm: Optional[str] = "default",
    ):
        """
        Initializes the SKFOptimizer.
        """

        self.skf = skf
        self._model_param = model_param
        self._param_space = param_space
        self._data = data
        self.detection_threshold = detection_threshold
        self.false_rate_threshold = false_rate_threshold
        self._max_timestep_to_detect = max_timestep_to_detect
        self._num_optimization_trial = num_optimization_trial
        self._grid_search = grid_search
        self._algorithm = algorithm
        self.skf_optim = None
        self.param_optim = None
        self._trial_count = 0
        self._backend = "optuna" if platform.system() == "Windows" else "ray"

    def objective(
        self,
        config,
        model_param: dict,
    ):
        """
        Returns a metric that is used for optimization.

        Returns:
            dict: Metric used for optimization.
        """

        skf = self.skf(
            config,
            model_param,
            self._data,
        )

        detection_rate = skf.metric_optim["detection_rate"]
        false_rate = skf.metric_optim["false_rate"]
        false_alarm_train = skf.metric_optim["false_alarm_train"]

        if (
            detection_rate < self.detection_threshold
            or false_rate > self.false_rate_threshold
            or false_alarm_train == "Yes"
        ):
            _metric = np.abs(self._param_space["slope"][1])  # upper bound of slope
        else:
            _metric = np.abs(config["slope"])

        metric = {}
        metric["metric"] = _metric
        metric["detection_rate"] = detection_rate
        metric["false_rate"] = false_rate
        metric["false_alarm_train"] = false_alarm_train
        return metric

    def optimize(self):
        """
        Run optimziation
        """

        if self._backend == "ray":
            self._ray_optimizer()
        elif self._backend == "optuna":
            self._optuna_optimizer()

    def get_best_model(self) -> SKF:
        """
        Retrieves the SKF instance initialized with the best parameters.

        Returns:
            Any: SKF instance corresponding to the optimal configuration.
        """
        return self.skf_optim

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

        # Parameter space
        search_config = self._ray_build_search_space()

        if self._grid_search:
            total_trials = 1
            for v in self._param_space.values():
                total_trials *= len(v)

            custom_logger = self._ray_progress_callback(total_samples=total_trials)

            optimizer_runner = tune.run(
                tune.with_parameters(
                    self.objective,
                    model_param=self._model_param,
                ),
                config=search_config,
                name="SKF_optimizer",
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
                    tune.with_parameters(
                        self.objective,
                        model_param=self._model_param,
                    ),
                    config=search_config,
                    search_alg=OptunaSearch(metric="metric", mode="min"),
                    name="SKF_optimizer",
                    num_samples=self._num_optimization_trial,
                    verbose=0,
                    raise_on_failed_trial=False,
                    callbacks=[custom_logger],
                )
            elif self._algorithm == "parallel":
                scheduler = ASHAScheduler(metric="metric", mode="min")
                optimizer_runner = tune.run(
                    tune.with_parameters(
                        self.objective,
                        model_param=self._model_param,
                    ),
                    config=search_config,
                    name="SKF_optimizer",
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

        # Get the optimal skf
        self.skf_optim = self.skf(
            self.param_optim,
            self._model_param,
            self._data,
        )

        # Print optimal parameters
        print("-----")
        print(f"Optimal parameters at trial #{best_sample_number}: {self.param_optim}")
        print("-----")

    def _ray_build_search_space(self) -> Dict:
        # Parameter space
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
                    if low < 0 or high < 0:
                        search_config[param_name] = tune.uniform(low, high)
                    else:
                        search_config[param_name] = tune.loguniform(low, high)
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
                params = trial.config
                self.trial_sample_map[trial.trial_id] = self.current_sample
                sample_str = f"{self.current_sample}/{self.total_samples}".rjust(
                    len(f"{self.total_samples}/{self.total_samples}")
                )
                print(
                    f"# {sample_str} - Metric: {result['metric']:.3f} - Detection rate: {result['detection_rate']:.2f} - False rate: {result['false_rate']:.2f} - False alarm in train: {result['false_alarm_train']} - Parameter: {params}"
                )

        return _Progress(total_samples)

    def _optuna_optimizer(self):
        """
        Optuna optimizer
        """

        if self._grid_search:
            sampler = optuna.samplers.GridSampler(self._param_space)
            self._num_optimization_trial = int(
                np.prod([len(v) for v in self._param_space.values()])
            )
        else:
            sampler = optuna.samplers.TPESampler()

        print("-----")
        print("SKF optimization starts")
        study = optuna.create_study(direction="minimize", sampler=sampler)

        study.optimize(
            self._optuna_objective,
            n_trials=self._num_optimization_trial,
            callbacks=[self._optuna_log_trial],
        )

        self.param_optim = study.best_params
        self.skf_optim = self.skf(
            self.param_optim,
            self._model_param,
            self._data,
        )

        print("-----")
        print(
            f"Optimal parameters at trial #{study.best_trial.number + 1}: "
            f"{self.param_optim}"
        )
        print(f"Best metric value: {study.best_value:.5f}")
        print("-----")

    def _optuna_objective(self, trial: optuna.Trial):
        """
        Objective function
        """

        param = self._optuna_build_search_space(trial)
        metric = self.objective(param, model_param=self._model_param)

        # Save extra info for callback
        trial.set_user_attr("detection_rate", metric["detection_rate"])
        trial.set_user_attr("false_rate", metric["false_rate"])
        trial.set_user_attr("false_alarm_train", metric["false_alarm_train"])

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
                    log_uniform = low > 0 and high > 0
                    param[name] = trial.suggest_float(name, low, high, log=log_uniform)
        return param

    def _optuna_log_trial(self, study: optuna.Study, trial: optuna.Trial):
        """
        Custom logging of trial progress.
        """

        self._trial_count += 1
        trial_id = f"{self._trial_count}/{self._num_optimization_trial}".rjust(
            len(f"{self._num_optimization_trial}/{self._num_optimization_trial}")
        )

        detection_rate = trial.user_attrs["detection_rate"]
        false_rate = trial.user_attrs["false_rate"]
        false_alarm_train = trial.user_attrs["false_alarm_train"]

        print(
            f"# {trial_id} - Metric: {trial.value:.5f} - Detection rate: {detection_rate:.2f} - "
            f"False rate: {false_rate:.2f} - False alarm in training data: {false_alarm_train} - Param: {trial.params}"
        )

        if trial.number == study.best_trial.number:
            print(
                f" -> New best trial #{trial.number + 1} with metric: {trial.value:.5f}"
            )
