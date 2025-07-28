import copy
import pandas as pd
from pytagi import Normalizer as normalizer
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pytagi.metric as metric
from statsmodels.tsa.seasonal import seasonal_decompose
from canari import (
    DataProcess,
    Model,
    SKF,
    plot_states,
    plot_data,
    plot_prediction,
    plot_skf_states,
)
from canari.component import (
    Exponential,
    WhiteNoise,
    Periodic,
    LocalTrend,
    Autoregression,
    LocalLevel,
)

df_raw = pd.read_csv(
    "/Users/michelwu/Desktop/Exponential component/2650F162.CSV",
    # "/Users/michelwu/Desktop/Exponential component/1700B042.CSV",
    # "/Users/michelwu/Desktop/Exponential component/0590P073.CSV",
    sep=";",  # Semicolon as delimiter
    quotechar='"',  # Double quotes as text qualifier
    engine="python",  # Python engine for complex cases
    na_values=[""],  # Treat empty strings as NaN
    skipinitialspace=True,  # Skip spaces after delimiter
    encoding="ISO-8859-1",
    parse_dates=["Date"],
    index_col="Date",
)

# Resample
df = df_raw[["Deplacements cumulatif X (mm)"]]
df = df
df = df.iloc[:]
df = df.resample("M").mean()

output_col = [0]
data_processor = DataProcess(
    data=df,
    train_split=0.7,
    validation_split=0.3,
    output_col=output_col,
    standardization=False,
)
train_data, validation_data, _, all_data = data_processor.get_splits()

# model localtrend seulement

localtrend1 = LocalTrend(mu_states=[2, -0.02], var_states=[1**2, 0.02**2], std_error=0)

# model expo+localtrend

exponential = Exponential(
    # std_error=0.0,
    mu_states=[0, 0.0028, 10.5, 0, 0],
    var_states=[0.1**2, 0.0001**2, 0.5**2, 0, 0],
)
localtrend2 = LocalTrend(
    mu_states=[1, -0.005], var_states=[0.2**2, 0.0005**2], std_error=0
)


# composants communs

periodic = Periodic(period=12, mu_states=[0, 0], var_states=[2, 2], std_error=0)


AR_process_error_var_prior = 0.5
var_W2bar_prior = 1
AR_process_error_var_prior = 1
var_W2bar_prior = 1
ar = Autoregression(
    mu_states=[-0.1, 0.7, 0, 0, 0, AR_process_error_var_prior],
    var_states=[
        6.36e-05,
        0.25,
        0,
        AR_process_error_var_prior,
        1e-6,
        var_W2bar_prior,
    ],
)

model1 = Model(localtrend1, ar, periodic)
mu_train_pred, std_train_pred, states = model1.filter(data=train_data)
mu_val_pred, std_val_pred, states = model1.forecast(data=validation_data)

fig, ax = plot_states(
    data_processor=data_processor,
    states=states,
    states_to_plot=["level"],
)
plot_data(
    data_processor=data_processor,
    plot_train_data=True,
    plot_test_data=False,
    plot_validation_data=True,
    sub_plot=ax[0],
)
plot_prediction(
    data_processor=data_processor,
    mean_validation_pred=mu_val_pred,
    std_validation_pred=std_val_pred,
    sub_plot=ax[0],
    color="k",
)
ax[0].scatter(
    data_processor.get_time("all"), data_processor.get_data("all"), color="red", s=2.5
)
ax[-1].set_xlabel("MM-DD")

model2 = Model(localtrend2, ar, periodic, exponential)
mu_train_pred, std_train_pred, states = model2.filter(data=train_data)
mu_val_pred, std_val_pred, states = model2.forecast(data=validation_data)

fig, ax = plot_states(
    data_processor=data_processor,
    states=states,
    states_to_plot=["scaled exp"],
)
plot_data(
    data_processor=data_processor,
    plot_train_data=True,
    plot_test_data=False,
    plot_validation_data=True,
    sub_plot=ax[-1],
)
plot_prediction(
    data_processor=data_processor,
    mean_validation_pred=mu_val_pred,
    std_validation_pred=std_val_pred,
    sub_plot=ax[-1],
    color="k",
)
ax[-1].scatter(
    data_processor.get_time("all"), data_processor.get_data("all"), color="red", s=2.5
)
ax[-1].set_xlabel("MM-DD")
plt.show()
