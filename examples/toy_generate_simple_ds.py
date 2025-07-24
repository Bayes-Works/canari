import numpy as np
from canari.component import LocalTrend, Periodic, WhiteNoise, Autoregression
from canari import (
    DataProcess,
    Model,
    plot_states,
)
import pandas as pd
from matplotlib import gridspec
import matplotlib.pyplot as plt

local_trend = LocalTrend(mu_states=[0.0, 0.0], var_states=[1e-12, 1e-12], std_error=0)
periodic = Periodic(period=52, mu_states=[1, 0], var_states=[1e-12, 1e-12])
ar = Autoregression(std_error=0.5, phi=0.9, mu_states=[-0.2], var_states=[1e-4])

model = Model(
    local_trend,
    periodic,
    ar
)

generated_ts, _, _, _ = model.generate_time_series(
    num_time_series=1,
    num_time_steps=1000,
)

# Save the generated time series to a csv file with column name as "syn_obs"
df_generated = pd.DataFrame(generated_ts[0], columns=["syn_obs"])
df_generated.to_csv("./data/toy_time_series/simple_syn_ar_std05_phi09_periodic.csv", index=False)

fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(1, 1)
ax0 = plt.subplot(gs[0])
ax0.plot(
    generated_ts[0],
    label="Generated time series",
    color="blue",
)
plt.show()