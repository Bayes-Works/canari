"""
Module: skf_optimizer

Optimizers for parameter tuning of SKF detection pipelines using Ray Tune.

Classes:

- SKFOptimizer: automated hyperparameter search using grid search or Bayesian optimization via Optuna/ASHAScheduler.
- CustomLogger: custom callback for progress logging during hyperparameter trials.

Usage example:

```python
from skf_optimizer import SKFOptimizer

def init_skf(config, model_params):
    # initialize your SKF detector with config and model parameters
    return MySKFDetector(**model_params, **config)

model_params = {...}
param_space = {"slope": [0.1, 2.0]}
data = {...}

optimizer = SKFOptimizer(
    initialize_skf=init_skf,
    model_param=model_params,
    param_space=param_space,
    data=data,
    detection_threshold=0.6,
    false_rate_threshold=0.1,
    num_optimization_trial=100,
)
optimizer.optimize()
best_model = optimizer.get_best_model()
```
"""

from ray import tune
from ray.tune import Callback, Stopper
from typing import Callable, Optional
from ray.tune.search.optuna import OptunaSearch
import numpy as np
from ray.tune.schedulers import ASHAScheduler
import signal

# Ignore segmentation fault signals
signal.signal(signal.SIGSEGV, lambda signum, frame: None)


class SKFOptimizer:
    """
    Parameter optimizer for SKF anomaly detection pipelines.

    This class runs hyperparameter search over the SKF detector configuration
    using Ray Tune. Supports both grid search and sampling-based algorithms
    (Optuna or ASHA scheduler).
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
        """
        Initializes the SKFOptimizer.

        Args:
            initialize_skf (Callable): Factory for creating an SKF detector instance.
            model_param (dict): Base parameters for the detector model.
            param_space (dict): Search space for hyperparameters.
            data (dict): Input data for synthetic anomaly detection.
            detection_threshold (float, optional): Minimum acceptable detection rate. Defaults to 0.5.
            false_rate_threshold (float, optional): Maximum acceptable false alarm rate. Defaults to 0.0.
            max_timestep_to_detect (int, optional): Max timesteps to allow detection. Defaults to None.
            num_synthetic_anomaly (int, optional): Number of anomalies to simulate. Defaults to 50.
            num_optimization_trial (int, optional): Number of trials for optimizer. Defaults to 50.
            grid_search (bool, optional): If True, perform exhaustive grid search. Defaults to False.
            algorithm (str, optional): Optimization algorithm ('default' for Optuna, 'parallel' for ASHA). Defaults to 'default'.

        Attributes:
            skf_optim: Best SKF instance after optimization.
            param_optim (dict): Best hyperparameter configuration.
        """

        self.initialize_skf = initialize_skf
        self.model_param = model_param
        self.param_space = param_space
        self.data = data
        self.detection_threshold = detection_threshold
        self.false_rate_threshold = false_rate_threshold
        self.max_timestep_to_detect = max_timestep_to_detect
        self.num_optimization_trial = num_optimization_trial
        self.num_synthetic_anomaly = num_synthetic_anomaly
        self.grid_search = grid_search
        self.algorithm = algorithm
        self.skf_optim = None
        self.param_optim = None

    def optimize(self):
        """
        Executes the hyperparameter search.

        Runs either grid search or sampling-based trials and logs progress.
        After completion, stores `param_optim` and `skf_optim`.
        """

        # Function for optimization
        def objective(
            config,
            model_param_param: dict,
        ):
            skf = self.initialize_skf(config, model_param_param)
            slope = config["slope"]

            detection_rate, false_rate, false_alarm_train = (
                skf.detect_synthetic_anomaly(
                    data=self.data,
                    num_anomaly=self.num_synthetic_anomaly,
                    slope_anomaly=slope,
                    max_timestep_to_detect=self.max_timestep_to_detect,
                )
            )

            if (
                detection_rate < self.detection_threshold
                or false_rate > self.false_rate_threshold
                or false_alarm_train == "Yes"
            ):
                metric = 2 + 5 * slope
            else:
                metric = detection_rate + 5 * np.abs(slope)

            tune.report(
                {
                    "metric": metric,
                    "detection_rate": detection_rate,
                    "false_rate": false_rate,
                    "false_alarm_train": false_alarm_train,
                }
            )

        # Parameter space
        search_config = {}
        if self.grid_search:
            total_trials = 1
            for param_name, values in self.param_space.items():
                search_config[param_name] = tune.grid_search(values)
                total_trials *= len(values)

            custom_logger = CustomLogger(total_samples=total_trials)
            optimizer_runner = tune.run(
                tune.with_parameters(
                    objective,
                    model_param_param=self.model_param,
                ),
                config=search_config,
                name="SKF_optimizer",
                num_samples=1,
                verbose=0,
                raise_on_failed_trial=False,
                callbacks=[custom_logger],
            )
        else:
            for param_name, values in self.param_space.items():
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

            # Run optimization
            custom_logger = CustomLogger(total_samples=self.num_optimization_trial)
            if self.algorithm == "default":
                optimizer_runner = tune.run(
                    tune.with_parameters(
                        objective,
                        model_param_param=self.model_param,
                    ),
                    config=search_config,
                    search_alg=OptunaSearch(metric="metric", mode="min"),
                    name="SKF_optimizer",
                    num_samples=self.num_optimization_trial,
                    verbose=0,
                    raise_on_failed_trial=False,
                    callbacks=[custom_logger],
                )
            elif self.algorithm == "parallel":
                scheduler = ASHAScheduler(metric="metric", mode="min")
                optimizer_runner = tune.run(
                    tune.with_parameters(
                        objective,
                        model_param_param=self.model_param,
                    ),
                    config=search_config,
                    name="SKF_optimizer",
                    num_samples=self.num_optimization_trial,
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
        self.skf_optim = self.initialize_skf(self.param_optim, self.model_param)

        # Print optimal parameters
        print("-----")
        print(f"Optimal parameters at trial #{best_sample_number}: {self.param_optim}")
        print("-----")

    def get_best_model(self):
        """
        Retrieves the SKF detector instance initialized with the best parameters.

        Returns:
            Any: SKF detector instance corresponding to the optimal configuration.
        """
        return self.skf_optim


class CustomLogger(Callback):
    """
    Ray Tune callback for logging trial progress.

    Logs the sample count and metrics for each trial as they complete.
    """

    def __init__(self, total_samples):
        self.total_samples = total_samples
        self.current_sample = 0
        self.trial_sample_map = {}

    def on_trial_result(self, iteration, trial, result, **info):
        """
        Called when a trial reports intermediate results.

        Args:
            iteration (int): Current iteration number.
            trial (Trial): Trial object.
            result (dict): Metrics reported by the trial.
            **info: Additional info.
        """

        self.current_sample += 1
        params = trial.config
        self.trial_sample_map[trial.trial_id] = self.current_sample
        sample_str = f"{self.current_sample}/{self.total_samples}".rjust(
            len(f"{self.total_samples}/{self.total_samples}")
        )
        print(
            f"# {sample_str} - Metric: {result['metric']:.3f} - Detection rate: {result['detection_rate']:.2f} - False rate: {result['false_rate']:.2f} - False alarm in train: {result['false_alarm_train']} - Parameter: {params}"
        )
