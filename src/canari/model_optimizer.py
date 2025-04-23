import signal
from typing import Callable, Optional
from ray import tune
from ray.tune import Callback
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from canari.data_process import DataProcess

# Ignore segmentation fault signals
signal.signal(signal.SIGSEGV, lambda signum, frame: None)


class ModelOptimizer:
    """
    Parameter optimization for model.py
    """

    def __init__(
        self,
        initialize_model: Callable,
        train: Callable,
        param_space: dict,
        data_processor: DataProcess,
        num_optimization_trial: Optional[int] = 50,
        grid_search: Optional[bool] = False,
        algorithm: Optional[str] = "default",
    ):
        self.initialize_model = initialize_model
        self.train = train
        self.param_space = param_space
        self.data_processor = data_processor
        self.num_optimization_trial = num_optimization_trial
        self.grid_search = grid_search
        self.algorithm = algorithm
        self.model_optim = None
        self.param_optim = None

    def optimize(self):
        """
        Optimization
        """

        # Function for optimization
        def objective(
            config,
        ):
            model = self.initialize_model(config)
            trained_model, *_ = self.train(
                model,
                self.data_processor,
            )
            tune.report({"metric": trained_model.early_stop_metric})

        # Parameter space
        search_config = {}
        if self.grid_search:
            total_trials = 1
            for param_name, values in self.param_space.items():
                search_config[param_name] = tune.grid_search(values)
                total_trials *= len(values)

            custom_logger = CustomLogger(total_samples=total_trials)
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
            for param_name, values in self.param_space.items():
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
            custom_logger = CustomLogger(total_samples=self.num_optimization_trial)
            if self.algorithm == "default":
                optimizer_runner = tune.run(
                    objective,
                    config=search_config,
                    search_alg=OptunaSearch(metric="metric", mode="min"),
                    name="Model_optimizer",
                    num_samples=self.num_optimization_trial,
                    verbose=0,
                    raise_on_failed_trial=False,
                    callbacks=[custom_logger],
                )
            elif self.algorithm == "parallel":
                scheduler = ASHAScheduler(metric="metric", mode="min")
                optimizer_runner = tune.run(
                    objective,
                    config=search_config,
                    name="Model_optimizer",
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

        # Get the optimal model
        self.model_optim = self.initialize_model(self.param_optim)

        # Print optimal parameters
        print("-----")
        print(f"Optimal parameters at trial #{best_sample_number}: {self.param_optim}")
        print("-----")

    def get_best_model(self):
        """
        Obtain optim model
        """
        return self.model_optim


class CustomLogger(Callback):
    def __init__(self, total_samples):
        self.total_samples = total_samples
        self.current_sample = 0
        self.trial_sample_map = {}

    def on_trial_result(self, iteration, trial, result, **info):
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
