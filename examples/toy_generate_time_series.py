import fire
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from src import (
    LocalTrend,
    Periodic,
    Autoregression,
    Model,
    plot_data,
    plot_prediction,
    plot_states,
)
from examples import DataProcess
from pytagi import exponential_scheduler
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from matplotlib import gridspec
from datetime import datetime, timedelta

model = Model(
    LocalTrend(mu_states=[0, 0], var_states=[1e-12, 1e-12], std_error=0),
    Periodic(period=52, mu_states=[0.5, 1], var_states=[1e-12, 1e-12]),
    Autoregression(std_error=0.1, phi=0.9, mu_states=[0], var_states=[0.08]),
)

generated_ts,_,_,_ = model.generate(num_time_series=1, num_time_steps=52*20)

# Save generated time series with the first row having string of 'syn_obs'
df = pd.DataFrame(generated_ts[0], columns=['syn_obs'])
df.to_csv("data/toy_time_series/synthetic_simple_autoregression_periodic.csv", index=False)

# Generate datetimes
start_date = '2000-01-01 12:00:00 PM'
interval_days = 7
start_datetime = datetime.strptime(start_date, '%Y-%m-%d %I:%M:%S %p')
date_series = np.array([start_datetime + timedelta(days=i * interval_days) for i in range(len(generated_ts[0]))])
df = pd.DataFrame(date_series, columns=['datetime'])
df.to_csv("data/toy_time_series/synthetic_simple_autoregression_periodic_datetime.csv", index=False)

# Plot generated time series
fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(1, 1)
ax0 = plt.subplot(gs[0])
for i in range(len(generated_ts)):
    ax0.plot(generated_ts[i])
ax0.set_title("Data generation")
plt.show()