import fire
import copy
import pandas as pd
import matplotlib.pyplot as plt
from pytagi import metric
from pytagi import Normalizer as normalizer
from canari.data_process import DataProcess
from canari.baseline_component import LocalTrend, LocalAcceleration
from canari.periodic_component import Periodic
from canari.white_noise_component import WhiteNoise
from canari.model import Model
from canari.SKF import SKF
from canari.model_optimizer import ModelOptimizer
from canari.SKF_optimizer import SKFOptimizer
from canari.data_visualization import (
    plot_data,
    plot_prediction,
    plot_skf_states,
    plot_states,
)
from ray import tune
from ray.tune.search.optuna import OptunaSearch

# Fix parameters grid search
sigma_v_fix = 0.015519087402266298
SKF_std_transition_error_fix = 0.0006733112773884772
SKF_norm_to_abnorm_prob_fix = 0.006047408738811242


def main(
    num_trial_optimization: int = 50,
    param_tune: bool = True,
):
    # Read data
    data_file = "./data/LTU014PIAEVA920.DAT"
    df_raw = pd.read_csv(
        data_file,
        sep=";",  # Semicolon as delimiter
        quotechar='"',
        engine="python",
        na_values=[""],  # Treat empty strings as NaN
        skipinitialspace=True,
        encoding="ISO-8859-1",
    )
    df = df_raw.iloc[1:, [6]]
    time = pd.to_datetime(df_raw.iloc[1:, 3])
    df.index = time
    df.index.name = "time"
    df.columns = ["Displacement"]
    df = df.resample("D").mean()

    # Data pre-processing
    output_col = [0]
    data_processor = DataProcess(
        data=df,
        train_split=0.9,
        validation_split=0.1,
        output_col=output_col,
        normalization=False,
    )
    (
        data_processor.train_data,
        data_processor.validation_data,
        data_processor.test_data,
        data_processor.all_data,
    ) = data_processor.get_splits()

    def objective(config):
        model = Model(
            LocalTrend(),
            Periodic(mu_states=[0.1, 0.1], var_states=[1, 1], period=365.24),
            Periodic(mu_states=[0.1, 0.1], var_states=[1, 1], period=162.12),
            WhiteNoise(std_error=1),
        )
        model.auto_initialize_baseline_states(
            data_processor.train_data["y"][0 : 365 * 2]
        )
        ab_model = Model(
            LocalAcceleration(),
            Periodic(mu_states=[0.1, 0.1], var_states=[1, 1], period=365.24),
            Periodic(mu_states=[0.1, 0.1], var_states=[1, 1], period=162.12),
            WhiteNoise(),
        )
        skf = SKF(
            norm_model=model,
            abnorm_model=ab_model,
            std_transition_error=config["std_transition_error"],
            norm_to_abnorm_prob=config["norm_to_abnorm_prob"],
            abnorm_to_norm_prob=1e-2,
            norm_model_prior_prob=0.99,
        )
        mu_preds, std_preds, filter_marginal_abnorm_prob, states = skf.filter(
            data=data_processor.all_data
        )

        obs = data_processor.get_data("all").flatten()
        log_lik = metric.log_likelihood(
            prediction=mu_preds,
            observation=obs,
            std=std_preds,
        )

        tune.report({"metric": log_lik})

    if param_tune:
        search_space = {
            "std_transition_error": tune.uniform(1e-6, 1e-3),
            "norm_to_abnorm_prob": tune.uniform(1e-6, 1e-3),
        }

        optimizer_runner = tune.run(
            objective,
            config=search_space,
            search_alg=OptunaSearch(metric="metric", mode="min"),
            name="Model_optimizer",
            num_samples=num_trial_optimization,
            verbose=1,
            raise_on_failed_trial=False,
        )

        param_optim = optimizer_runner.get_best_config(metric="metric", mode="min")

        print("Best config:", param_optim)

    else:
        config = {
            "std_transition_error": SKF_std_transition_error_fix,
            "norm_to_abnorm_prob": SKF_norm_to_abnorm_prob_fix,
        }
        objective(config)


if __name__ == "__main__":
    fire.Fire(main)
