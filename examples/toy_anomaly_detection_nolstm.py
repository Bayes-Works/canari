import fire
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from canari import (
    DataProcess,
    Model,
    SKF,
    plot_skf_states,
)
from canari.component import (
    LocalTrend,
    LocalAcceleration,
    WhiteNoise,
    Periodic,
    Autoregression,
)

# Read data
data_file = "./data/toy_time_series/synthetic_autoregression_periodic.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
data_file_time = "./data/toy_time_series/synthetic_autoregression_periodic_datetime.csv"
time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(time_series[0])
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values"]

# Add synthetic anomaly to data
time_anomaly = 400
AR_stationary_var = 5**2 / (1 - 0.9**2)
anomaly_magnitude = -(np.sqrt(AR_stationary_var) * 1) / 50
for i in range(time_anomaly, len(df_raw)):
    df_raw.values[i] += anomaly_magnitude * (i - time_anomaly)

# Data pre-processing
output_col = [0]
data_processor = DataProcess(
    data=df_raw,
    train_split=1,
    output_col=output_col,
    normalization=False,
)
_, _, _, all_data = data_processor.get_splits()


def main(
    case: int = 4,
):

    # Components
    sigma_v = np.sqrt(1e-6)
    local_trend = LocalTrend(mu_states=[5, 0.0], var_states=[1e-12, 1e-12])
    local_acceleration = LocalAcceleration(
        mu_states=[5, 0.0, 0.0], var_states=[1e-12, 1e-4, 1e-4]
    )
    periodic = Periodic(period=52, mu_states=[5 * 5, 0], var_states=[1e-12, 1e-12])
    noise = WhiteNoise(std_error=sigma_v)

    # Different cases for ar components
    if case == 1:
        # Case 1: regular ar, with process error and phi provided
        ar = Autoregression(
            std_error=5, phi=0.9, mu_states=[-0.0621], var_states=[6.36e-05]
        )

    elif case == 2:
        # Case 2: ar with process error provided, learn phi online. It should converge to ~0.9
        ar = Autoregression(
            std_error=5, mu_states=[-0.0621, 0.5, 0], var_states=[6.36e-05, 0.25, 0]
        )

    elif case == 3:
        # Case 3: ar with phi provided, learn process error online. W2bar (variance of process error) should converge to ~25.
        ar = Autoregression(
            phi=0.9,
            mu_states=[-0.0621, 0, 0, 100],
            var_states=[6.36e-05, 100, 1e-6, 100],
        )

    elif case == 4:
        # Case 4: Fully online ar, learn both phi and process error online. phi should converge to ~0.9, W2bar should converge to ~25.
        ar = Autoregression(
            mu_states=[-0.0621, 0.5, 0, 0, 0, 100],
            var_states=[6.36e-05, 0.25, 0, 100, 1e-6, 100],
        )

    # Normal model
    model = Model(
        local_trend,
        periodic,
        ar,
        noise,
    )

    #  Abnormal model
    ab_model = Model(
        local_acceleration,
        periodic,
        ar,
        noise,
    )

    # Switching Kalman filter
    skf = SKF(
        norm_model=model,
        abnorm_model=ab_model,
        std_transition_error=1e-3,
        norm_to_abnorm_prob=1e-4,
        abnorm_to_norm_prob=1e-4,
        norm_model_prior_prob=0.99,
    )

    # # # Anomaly Detection
    filter_marginal_abnorm_prob, states = skf.filter(data=all_data)
    smooth_marginal_abnorm_prob, states = skf.smoother(data=all_data)

    #  Plot
    marginal_abnorm_prob_plot = filter_marginal_abnorm_prob
    fig, ax = plot_skf_states(
        data_processor=data_processor,
        states=states,
        states_type="prior",
        model_prob=marginal_abnorm_prob_plot,
        # states_to_plot=[
        #     "local level",
        #     "local trend",
        #     "periodic 1",
        #     "autoregression",
        #     "phi",
        #     "AR_error",
        #     "W2",
        #     "W2bar",
        # ],
        color="b",
    )
    fig.suptitle("SKF hidden states", fontsize=10, y=1)
    plt.show()


if __name__ == "__main__":
    fire.Fire(main)
