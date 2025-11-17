"""
This module automates the search for optimal hyperparameters of a
:class:`~canari.skf.SKF` instance by leveraging the Optuna
external library.
"""

from typing import Callable, Dict, Optional
import numpy as np

import signal
from ray import tune
from ray.tune import Callback
from typing import Callable, Optional
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
import nevergrad as ng
from ray.tune.search.nevergrad import NevergradSearch
from canari import SKF
from scipy.stats import norm, lognorm
from fractions import Fraction
import matplotlib.pyplot as plt

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
        back_end(str, optional): "ray" or "optuna". Using the external library Ray or Optuna
                                    for optimization. Default to "ray".

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
        algorithm: Optional[str] = "default",  # "default", "BOHB"
        back_end: Optional[str] = "ray",
        mode: Optional[str] = "max",
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
        self._backend = back_end
        self._mode = mode

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
        anm_magnitude = skf.metric_optim["anomaly_magnitude"]

        mean = 0.5  # mean of normal
        std_dev = 0.05  # std deviation of normal

        # metric
        j1 = norm.cdf(detection_rate, loc=mean, scale=std_dev)
        j2 = 1 - lognorm.cdf(false_rate, s=0.2, scale=0.1)
        j3 = 1 - lognorm.cdf(anm_magnitude, s=0.2, scale=0.4)
        # j3 = 1 - lognorm.cdf(anm_magnitude, s=0.3, scale=0.1)
        _metric = j1 * j2 * j3

        metric = {}
        metric["metric"] = _metric
        metric["detection_rate"] = detection_rate
        metric["false_rate"] = false_rate
        metric["false_alarm_train"] = anm_magnitude

        return metric

    def optimize(self):
        """
        Run optimziation
        """

        if self._backend == "ray":
            self._ray_optimizer()
        # elif self._backend == "optuna":
        #     self._optuna_optimizer()

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
                sampler = optuna.samplers.TPESampler(
                    n_startup_trials=40,
                    multivariate=True,
                    group=True,
                )
                optimizer_runner = tune.run(
                    tune.with_parameters(
                        self.objective,
                        model_param=self._model_param,
                    ),
                    config=search_config,
                    search_alg=OptunaSearch(
                        metric="metric",
                        mode=self._mode,
                        sampler=sampler,
                    ),
                    name="SKF_optimizer",
                    num_samples=self._num_optimization_trial,
                    verbose=0,
                    raise_on_failed_trial=False,
                    callbacks=[custom_logger],
                )
            elif self._algorithm == "parallel":
                scheduler = ASHAScheduler(metric="metric", mode=self._mode)
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
            elif self._algorithm == "Nevergrad":
                search_alg = NevergradSearch(
                    optimizer=ng.optimizers.OnePlusOne,  # or any other NG optimizer
                    metric="metric",
                    mode=self._mode,
                )

                optimizer_runner = tune.run(
                    tune.with_parameters(
                        self.objective,
                        model_param=self._model_param,
                    ),
                    config=search_config,
                    search_alg=search_alg,
                    name="SKF_optimizer",
                    num_samples=self._num_optimization_trial,
                    verbose=0,
                    raise_on_failed_trial=False,
                    callbacks=[custom_logger],
                )

        # Get the optimal parameters
        self.param_optim = optimizer_runner.get_best_config(
            metric="metric", mode=self._mode
        )
        best_trial = optimizer_runner.get_best_trial(metric="metric", mode=self._mode)
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
                    f"# {sample_str} - Metric: {result['metric']:.8f} - Detection rate: {result['detection_rate']:.8f} - False rate: {result['false_rate']:.8f} - False alarm in train: {result['false_alarm_train']} - Parameter: {params}"
                )

        return _Progress(total_samples)
