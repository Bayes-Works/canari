import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from matplotlib import gridspec
from canari import (
    DataProcess,
    Model,
    plot_data,
    plot_prediction,
    plot_states,
)
from canari.component import LocalTrend, LstmNetwork, Autoregression, Periodic

# Set numpy seeds
# np.random.seed(6)
np.random.seed(895)

# Define SSM
model = Model(
    LocalTrend(
        mu_states=[0, 0],
        var_states=[1e-12, 1e-12],
        std_error=0,
    ),  
    Periodic(period=52, mu_states=[0, 5 * 5], var_states=[1e-12, 1e-12]),
    Periodic(period=13, mu_states=[0, 10], var_states=[1e-12, 1e-12]),
    Autoregression(
        std_error=5, phi=0.5, mu_states=[-0.0621], var_states=[6.36e-05]
    ),
)
num_time_steps = 52 * 12
gen_ts, _, _, _ = model.generate_time_series(num_time_series=1,
                                    num_time_steps=num_time_steps,
                                    )

# Generate timestamps beginning from 2011-02-06  12:00:00 AM, interval one week, and same num_time_steps
start_time = pd.Timestamp('2013-02-06 00:00:00')
# time_stamps = pd.date_range(start=start_time, periods=num_time_steps, freq='W')
# Generate time_stamps with format yyyy-mm-dd hh:mm:ss
time_stamps = pd.date_range(start=start_time, periods=num_time_steps, freq='W').strftime('%Y-%m-%d %H:%M:%S')

print(gen_ts[0])
print(time_stamps)

# Save start_time and time_stamps in a csv with two columns timestamp and value
df = pd.DataFrame({
    'timestamp': time_stamps,
    'value': gen_ts[0]
})
df.to_csv('data/toy_time_series/syn_data_anmtype_simple_phi05_v2.csv', index=False)

# Plot gen_ts
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(gen_ts[0], label='Generated Time Series')
ax.set_title('Generated Time Series from State Space Model')
ax.set_xlabel('Time Steps')
ax.set_ylabel('Value')
ax.legend()
plt.show()