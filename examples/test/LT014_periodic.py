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
sigma_v_fix = 1
SKF_std_transition_error_fix = 1e-5
SKF_norm_to_abnorm_prob_fix = 1e-5

# sigma_v_fix = 1
# SKF_std_transition_error_fix = 1.461560693168833e-06
# SKF_norm_to_abnorm_prob_fix = 0.00014170862558977718


def main(
    num_trial_optimization: int = 20,
    param_tune: bool = False,
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

    model = Model(
        LocalTrend(),
        Periodic(mu_states=[0.1, 0.1], var_states=[1, 1], period=365.24),
        Periodic(mu_states=[0.1, 0.1], var_states=[1, 1], period=162.12),
        WhiteNoise(std_error=sigma_v_fix),
    )
    model.auto_initialize_baseline_states(data_processor.train_data["y"][0:365])
    ab_model = Model(
        LocalAcceleration(),
        Periodic(mu_states=[0.1, 0.1], var_states=[1, 1], period=365.24),
        Periodic(mu_states=[0.1, 0.1], var_states=[1, 1], period=162.12),
        WhiteNoise(),
    )
    skf = SKF(
        norm_model=model,
        abnorm_model=ab_model,
        std_transition_error=SKF_std_transition_error_fix,
        norm_to_abnorm_prob=SKF_norm_to_abnorm_prob_fix,
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

    fig, ax = plot_skf_states(
        data_processor=data_processor,
        states=states,
        states_to_plot=["local level", "local trend", "periodic 1", "white noise"],
        model_prob=filter_marginal_abnorm_prob,
        normalization=False,
        color="b",
    )
    fig.suptitle("SKF hidden states", fontsize=10, y=1)
    plt.show()


if __name__ == "__main__":
    fire.Fire(main)
