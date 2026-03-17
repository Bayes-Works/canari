import copy
from scipy.optimize import minimize
import scipy
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
from pytagi import Normalizer as normalizer
import pytagi.metric as metric
from canari import (
    DataProcess,
    Model,
    plot_states,
    plot_data,
    plot_prediction,
)
from canari.component import (
    Exponential,
    WhiteNoise,
    LocalTrend,
    Periodic,
    LocalLevel,
    Autoregression,
    LocalAcceleration,
    LstmNetwork
)
from matplotlib.lines import Line2D
df_raw = pd.read_csv(
    "/Users/michelwu/Desktop/Exp DAT/reel_data/LTU014PIAEVA920.DAT",
    sep=";",  # Semicolon as delimiter
    quotechar='"',  # Double quotes as text qualifier
    engine="python",  # Python engine for complex cases
    na_values=[""],  # Treat empty strings as NaN
    skipinitialspace=True,  # Skip spaces after delimiter
    encoding="ISO-8859-1",
    parse_dates=["Date"],
    index_col="Date",
)

df = df_raw[["Deplacements cumulatif X (mm)"]]
df = df.iloc[:]
mask = ~np.isnan(df)
df = df[mask].resample("W").mean()



# date_1 = "2010-08-29"
date_1 = "2010-07-04"
date_2 = "2014-10-27"
mask1 = df.index < date_1
df_part1 = df[mask1]
# df_part1.iloc[370:]=np.nan
mask2 = (df.index >= date_1) & (df.index < date_2)
df_part2 = df[mask2]
# print(df_part2.iloc[:9])
# df_part2.iloc[:9]=np.nan
# mask_data = df.index < date_2
# df_part = df[mask_data]

first_year = df_part1.index.min().year
last_year = df_part1.index.max().year
years = [str(year) for year in range(first_year, last_year - 4)]
validation_start_str = df_part1.loc[str(last_year - 1)].index[0].strftime("%Y-%m-%d")
last_date_str=df_part1.index[-1].strftime("%Y-%m-%d")
# test_start_str = df.loc[str(last_year - 1)].index[0].strftime("%Y-%m-%d")

data_processor1 = DataProcess(
    data=df_part1,
    train_start=df_part1.loc[years[0]].index[0].strftime("%Y-%m-%d"),
    validation_start=validation_start_str,
    validation_end=last_date_str,
    test_start=last_date_str,
    output_col=[0],
    standardization=False,
)


train_data, validation_data, test_data, all_data = data_processor1.get_splits()
df_train=pd.DataFrame(index=train_data["time"], data={'y':train_data["y"].flatten()})

df.index.name="date"

expo=Exponential()
local_level=LocalLevel()
ar=Autoregression(phi=0,mu_states=[0,0,0,0])
lstm=LstmNetwork(look_back_len=52,
        num_features=1,
        infer_len=52 * 3,
        num_layer=1,
        num_hidden_unit=40,
        device="cpu",
        manual_seed=42,
        )
model = Model(expo,ar,lstm,local_level)
model.auto_initialize_comp(data_training=df_train,ratio_training=0.9)
output_col = [0]
num_epoch = 50
for epoch in range(num_epoch):
    (mu_validation_preds, std_validation_preds, states) = model.lstm_train(
        train_data=train_data,
        validation_data=validation_data,
    )

    # Unstandardize the predictions
    mu_validation_preds = normalizer.unstandardize(
        mu_validation_preds,
        data_processor1.scale_const_mean[output_col],
        data_processor1.scale_const_std[output_col],
    )
    std_validation_preds = normalizer.unstandardize_std(
        std_validation_preds,
        data_processor1.scale_const_std[output_col],
    )

    # Calculate the log-likelihood metric
    validation_obs = data_processor1.get_data("validation").flatten()
    mse = metric.mse(mu_validation_preds, validation_obs)

    # Early-stopping
    model.early_stopping(evaluate_metric=mse, current_epoch=epoch, max_epoch=num_epoch)
    if epoch == model.optimal_epoch:
        mu_validation_preds_optim = mu_validation_preds
        std_validation_preds_optim = std_validation_preds
        states_optim = copy.copy(
            states
        )  # If we want to plot the states, plot those from optimal epoch

    if model.stop_training:
        break


print(f"Optimal epoch       : {model.optimal_epoch}")
print(f"Validation MSE      :{model.early_stop_metric: 0.4f}")

print(model.mu_states)