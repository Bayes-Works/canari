"""
This module automates the search for optimal hyperparameters of a
:class:`~canari.model.Model` instance by leveraging external libraries.
"""

from typing import Callable, Dict, Optional
import signal
from ray import tune
from ray.tune import Callback
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.sample import Domain
import optuna
from canari import Model

signal.signal(signal.SIGSEGV, lambda signum, frame: None)


class Optimizer:
    """
    Optimize hyperparameters for :class:`~canari.model.Model` using the Ray Tune
    external library using the metric :attr:`~canari.model.Model.metric_optim`.

    Args:
        model (Callable):
            Function that returns a model instance given a model configuration.
        param (Dict[str, list]):
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
        back_end(str, optional): "ray". Using the external library Ray for optimization.

    Attributes:
        model_optim :
            The best model instance initialized with optimal parameters after running optimize().
        param_optim (Dict):
            The best hyperparameter configuration found during optimization.
    """

    def __init__(
        self,
        model: Callable,
        param: dict,
        model_input: Optional[dict] = None,
        num_optimization_trial: Optional[int] = 50,
        grid_search: Optional[bool] = False,
        mode: Optional[str] = "min",
        algorithm: Optional[str] = "TPE",  # "TPE" or "random"
        back_end: Optional[str] = "ray",
        num_startup_trials: Optional[int] = 20,
    ):
        """
        Initialize the ModelOptimizer.
        """

        self.model = model
        self.model_optim = None
        self.param_optim = None
        self._param = param
        self._model_input = model_input
        self._num_optimization_trial = num_optimization_trial
        self._grid_search = grid_search
        self._mode = mode
        self._trial_count = 0
        self._algorithm = algorithm
        self._backend = back_end
        self._num_startup_trials = num_startup_trials

    def objective(self, config: Dict) -> Dict:
        """
        Returns a metric that is used for optimization

        Returns:
            dict: Metric used for optimization.
        """
        if self._model_input is None:
            result = self.model(config)
        else:
            result = self.model(config, self._model_input)

        if not isinstance(result, tuple):
            result = (result,)
        trained_model, *_ = result

        _metric = trained_model.metric_optim
        _print_metric = trained_model.print_metric

        metric = {}
        metric["metric"] = _metric
        metric["print_metric"] = _print_metric
        return metric

    def optimize(self):
        """
        Run optimziation
        """

        if self._backend == "ray":
            self._ray_optimizer()

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
            for v in self._param.values():
                total_trials *= len(v)
            custom_logger = self._ray_progress_callback(total_samples=total_trials)

            optimizer_runner = tune.run(
                self.objective,
                config=search_config,
                num_samples=1,
                verbose=0,
                raise_on_failed_trial=False,
                callbacks=[custom_logger],
            )
        else:
            custom_logger = self._ray_progress_callback(
                total_samples=self._num_optimization_trial
            )
            if self._algorithm == "TPE":
                sampler = optuna.samplers.TPESampler(
                    n_startup_trials=self._num_startup_trials,
                    multivariate=True,
                    group=True,
                )
                optimizer_runner = tune.run(
                    self.objective,
                    config=search_config,
                    search_alg=OptunaSearch(
                        metric="metric", mode=self._mode, sampler=sampler
                    ),
                    num_samples=self._num_optimization_trial,
                    verbose=0,
                    raise_on_failed_trial=False,
                    callbacks=[custom_logger],
                )
            elif self._algorithm == "random":
                scheduler = ASHAScheduler(metric="metric", mode=self._mode)
                optimizer_runner = tune.run(
                    self.objective,
                    config=search_config,
                    num_samples=self._num_optimization_trial,
                    scheduler=scheduler,
                    verbose=0,
                    raise_on_failed_trial=False,
                    callbacks=[custom_logger],
                )
            else:
                raise ValueError("algorithm must be 'TPE' or 'random'")

        # Best params & model
        self.param_optim = optimizer_runner.get_best_config(
            metric="metric", mode=self._mode
        )
        best_trial = optimizer_runner.get_best_trial(metric="metric", mode=self._mode)
        best_sample_number = custom_logger.trial_sample_map.get(
            best_trial.trial_id, "Unknown"
        )

        if self._model_input is None:
            result = self.model(self.param_optim)
        else:
            result = self.model(self.param_optim, self._model_input)

        if not isinstance(result, tuple):
            result = (result,)
        best_model, *_ = result

        self.model_optim = best_model

        print("-----")
        print(
            f"Optimal parameters at trial #{best_sample_number}. Best metric: {best_model.metric_optim:.4f}. Best print metric: {best_model.print_metric}. Best param: {self.param_optim}."
        )
        print("-----")

    def _ray_build_search_space(self) -> Dict:
        """
        Convert param to Ray Tune search space objects.
        """
        search_config = {}
        for param_name, values in self._param.items():
            if self._grid_search:
                search_config[param_name] = tune.grid_search(values)
                print("here grid search")
                continue

            if isinstance(values, Domain):
                search_config[param_name] = values
                continue

            if isinstance(values, list) and len(values) == 2:
                low, high = values

                if isinstance(low, int) and isinstance(high, int):
                    search_config[param_name] = tune.randint(low, high + 1)
                    continue

                if isinstance(low, float) and isinstance(high, float):
                    search_config[param_name] = tune.uniform(low, high)
                    continue

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
                print_metric = result["print_metric"]

                self.trial_sample_map[trial.trial_id] = self.current_sample

                width = len(f"{self.total_samples}/{self.total_samples}")
                sample_str = f"{self.current_sample}/{self.total_samples}".rjust(width)
                if print_metric is None:
                    print(
                        f"# {sample_str} - Metric: {metric:.3f} - Parameter: {params}"
                    )
                else:
                    print(
                        f"# {sample_str} - Metric: {metric:.3f} - Print metric: {print_metric} - Parameter: {params}"
                    )

        return _Progress(total_samples)
