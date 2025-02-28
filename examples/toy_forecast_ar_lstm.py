import fire
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from src import (
    LocalLevel,
    LocalTrend,
    LocalAcceleration,
    LstmNetwork,
    Periodic,
    Autoregression,
    WhiteNoise,
    Model,
    plot_data,
    plot_prediction,
    plot_states,
)
from examples import DataProcess
from pytagi import exponential_scheduler
import pytagi.metric as metric
from pytagi import Normalizer as normalizer


# # Read data
data_file = "./data/toy_time_series/synthetic_autoregression_periodic.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)

data_file_time = "./data/toy_time_series/synthetic_autoregression_periodic_datetime.csv"
time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(time_series[0])
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values"]

# # Skip resampling data
df = df_raw

# Define parameters
output_col = [0]
num_epoch = 50

data_processor = DataProcess(
    data=df,
    train_split=0.8,
    validation_split=0.2,
    output_col=output_col,
)

train_data, validation_data, test_data, normalized_data = data_processor.get_splits()

sigma_v = np.sqrt(1e-6)/(data_processor.norm_const_std[output_col].item()+1e-10)
noise = WhiteNoise(std_error=sigma_v)

def main(
    case: int = 4,
):
    # Define AR
    # stationary AR std in standardized space: 1.103733186675304
    if case == 1:
        # Case 1
        AR = Autoregression(std_error=0.23146312, phi=0.9, mu_states=[0], var_states=[1e-06])

    elif case == 2:
        # Case 2
        AR = Autoregression(std_error=0.23146312, mu_states=[0, 0, 0], var_states=[1e-06, 0.25, 0])

    elif case == 3:
        # Case 3
        AR_process_error_var_prior = 1
        var_W2bar_prior = 1
        AR = Autoregression(phi=0.9, mu_states=[0, 0, 0, AR_process_error_var_prior],var_states=[1e-06, AR_process_error_var_prior, 1e-6, var_W2bar_prior])

    elif case == 4:
        # Case 4
        AR_process_error_var_prior = 1
        var_W2bar_prior = 1
        AR = Autoregression(mu_states=[0, 0, 0, 0, 0, AR_process_error_var_prior],var_states=[1e-06, 0.25, 0, AR_process_error_var_prior, 1e-6, var_W2bar_prior])

    model = Model(
        LocalTrend(mu_states=[-0.00902307, 0.0], var_states=[1e-12, 1e-12], std_error=0), # True baseline values
        # LocalTrend(),
        LstmNetwork(
            look_back_len=52,
            num_features=1,
            num_layer=1,
            num_hidden_unit=50,
            device="cpu",
        ),
        AR,
        noise,
    )
    # model.auto_initialize_baseline_states(train_data["y"][0:52*6])

    # Training
    for epoch in range(num_epoch):

        mu_validation_preds, std_validation_preds, states = model.lstm_train(
            train_data=train_data,
            validation_data=validation_data,
        )

        # # Unstandardize the predictions
        # mu_validation_preds = normalizer.unstandardize(
        #     mu_validation_preds,
        #     data_processor.norm_const_mean[output_col],
        #     data_processor.norm_const_std[output_col],
        # )
        # std_validation_preds = normalizer.unstandardize_std(
        #     std_validation_preds,
        #     data_processor.norm_const_std[output_col],
        # )

        # Calculate the log-likelihood metric
        mse = metric.mse(
            mu_validation_preds, data_processor.validation_data[:, output_col].flatten()
        )

        # Early-stopping
        model.early_stopping(evaluate_metric=mse, mode="min")

        if epoch == model.optimal_epoch:
                mu_validation_preds_optim = mu_validation_preds.copy()
                std_validation_preds_optim = std_validation_preds.copy()
                states_optim = copy.copy(states)
        if model.stop_training:
            break

    print(f"Optimal epoch       : {model.optimal_epoch}")
    print(f"Validation MSE      :{model.early_stop_metric: 0.4f}")

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
    plt.legend()
    plot_states(
        data_processor=data_processor,
        states=model.states,
        states_type="prior",
        states_to_plot=['local level'],
        sub_plot=ax,
    )
    plot_states(
        data_processor=data_processor,
        states=model.states,
        states_type="prior",
    )
    print(f"States prior at the last step: ")
    print(model.states.mu_prior[-1])
    plt.show()

if __name__ == "__main__":
    fire.Fire(main)
