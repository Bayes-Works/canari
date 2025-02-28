import fire
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src import (
    LocalTrend,
    LocalAcceleration,
    Periodic,
    Autoregression,
    WhiteNoise,
    Model,
    plot_data,
    plot_prediction,
    plot_states,
)
from examples import DataProcess


# # Read data
data_file = "./data/toy_time_series/synthetic_autoregression_periodic.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
data_file_time = "./data/toy_time_series/synthetic_autoregression_periodic_datetime.csv"
time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(time_series[0])
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values"]

# Data pre-processing
all_data = {}
all_data["y"] = df_raw.values

# Split into train and test
output_col = [0]
data_processor = DataProcess(
    data=df_raw,
    train_split=0.8,
    validation_split=0.2,
    output_col=output_col,
    normalization=False,
)
train_data, validation_data, _, _ = data_processor.get_splits()


def main(
    case: int = 4,
):
    # Components
    sigma_v = np.sqrt(1e-6)
    local_trend = LocalTrend(mu_states=[5, 0.0], var_states=[1e-1, 1e-6], std_error=0)
    periodic = Periodic(period=52, mu_states=[5 * 5, 0], var_states=[1e-12, 1e-12])
    noise = WhiteNoise(std_error=sigma_v)

    # Different cases for AR components
    if case == 1:
        # Case 1: regular AR, with process error and phi provided
        AR = Autoregression(
            std_error=5, phi=0.9, mu_states=[-0.0621], var_states=[6.36e-05]
        )
    elif case == 2:
        # Case 2: AR with process error provided, learn phi online. It should converge to ~0.9
        AR = Autoregression(
            std_error=5, mu_states=[-0.0621, 0.5, 0], var_states=[6.36e-05, 0.25, 0]
        )
    elif case == 3:
        # Case 3: AR with phi provided, learn process error online. W2bar (variance of process error) should converge to ~25.
        AR_process_error_var_prior = 100
        var_W2bar_prior = 100
        AR = Autoregression(
            phi=0.9,
            mu_states=[-0.0621, 0, 0, AR_process_error_var_prior],
            var_states=[6.36e-05, AR_process_error_var_prior, 1e-6, var_W2bar_prior],
        )
    elif case == 4:
        # Case 4: Fully online AR, learn both phi and process error online. phi should converge to ~0.9, W2bar should converge to ~25.
        AR_process_error_var_prior = 100
        var_W2bar_prior = 100
        AR = Autoregression(
            mu_states=[-0.0621, 0.5, 0, 0, 0, AR_process_error_var_prior],
            var_states=[
                6.36e-05,
                0.25,
                0,
                AR_process_error_var_prior,
                1e-6,
                var_W2bar_prior,
            ],
        )

    # Normal model
    model = Model(
        local_trend,
        periodic,
        AR,
        noise,
    )

    # # #
    model.filter(data=train_data)
    model.smoother(data=train_data)
    mu_validation_preds, std_validation_preds = model.forecast(validation_data)

    #  Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_data(
        data_processor=data_processor,
        normalization=True,
        plot_column=output_col,
        validation_label="y",
    )
    plot_prediction(
        data_processor=data_processor,
        mean_validation_pred=mu_validation_preds,
        std_validation_pred=std_validation_preds,
        validation_label=[r"$\mu$", f"$\pm\sigma$"],
    )

    # #  Plot
    plot_states(
        data_processor=data_processor,
        states=model.states,
        states_type="prior",
    )
    # plot_states(
    #     data_processor=data_processor,
    #     states=model.states,
    #     states_type="posterior",
    # )
    # plot_states(
    #     data_processor=data_processor,
    #     states=model.states,
    #     states_type="smooth",
    # )
    plt.show()


if __name__ == "__main__":
    fire.Fire(main)
