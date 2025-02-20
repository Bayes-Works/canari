from ray import tune
from ray.tune import Callback
from typing import Callable, Optional
from ray.tune.search.optuna import OptunaSearch
from examples.data_process import DataProcess


class CustomLogger(Callback):
    def __init__(self, total_samples):
        self.total_samples = total_samples
        self.current_sample = 0
        self.trial_sample_map = {}

    def on_trial_result(self, iteration, trial, result, **info):
        self.current_sample += 1
        params = trial.config
        metric = result["metric"]

        self.trial_sample_map[trial.trial_id] = self.current_sample
        sample_str = f"{self.current_sample}/{self.total_samples}".rjust(
            len(f"{self.total_samples}/{self.total_samples}")
        )
        print(
            f"# {sample_str} - Metric: {result['metric']:.3f} - Detection rate: {result['detection_rate']:.2f} - False rate: {result['false_rate']:.2f} - Parameter: {params}"
        )


class SKFOptimizer:
    """
    Parameter optimization for model.py
    """

    def __init__(
        self,
        initialize_skf: Callable,
        model: dict,
        param_space: dict,
        data_processor: DataProcess,
        detection_threshold: Optional[float] = 0.5,
        false_rate_threshold: Optional[float] = 0.0,
        num_synthetic_anomaly: Optional[int] = 50,
        num_optimization_trial: Optional[int] = 50,
        grid_search: Optional[bool] = False,
    ):
        self.initialize_skf = initialize_skf
        self.model = model
        self.param_space = param_space
        self.data_processor = data_processor
        self.detection_threshold = detection_threshold
        self.false_rate_threshold = false_rate_threshold
        self.num_optimization_trial = num_optimization_trial
        self.num_synthetic_anomaly = num_synthetic_anomaly
        self.grid_search = grid_search
        self.skf_optim = None
        self.param_optim = None

    def optimize(self):
        """
        Optimization
        """

        # Function for optimization
        def objective(
            config,
            model_param: dict,
        ):
            skf = self.initialize_skf(config, model_param)
            slope = config["slope"]

            detection_rate, false_rate = skf.detect_synthetic_anomaly(
                data=self.data_processor.train_split,
                num_anomaly=self.num_synthetic_anomaly,
                slope_anomaly=slope,
            )

            if (
                detection_rate < self.detection_threshold
                or false_rate > self.false_rate_threshold
            ):
                metric = 2 + 5 * slope
            else:
                metric = detection_rate + 5 * slope

            tune.report(
                {
                    "metric": metric,
                    "detection_rate": detection_rate,
                    "false_rate": false_rate,
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
                    model_param=self.model,
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
            optimizer_runner = tune.run(
                tune.with_parameters(
                    objective,
                    model_param=self.model,
                ),
                config=search_config,
                search_alg=OptunaSearch(metric="metric", mode="min"),
                name="SKF_optimizer",
                num_samples=self.num_optimization_trial,
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
        self.skf_optim = self.initialize_skf(self.param_optim, self.model)

        # Print optimal parameters
        print("-----")
        print(f"Optimal parameters at trial #{best_sample_number}: {self.param_optim}")
        print("-----")

    def get_best_model(self):
        """
        Obtain optim model
        """
        return self.skf_optim
