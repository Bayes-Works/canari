import numpy as np
import stumpy

# Generate a time series of a sine wave
num_ts = 1000
time_series = np.sin(np.linspace(0, 20 * np.pi, num_ts))

# Add anomaly
anm_index = 600
# LT anomaly
anm_mag = -0.2/52
# anm_mag = 0
anm_baseline = np.arange(0, num_ts-anm_index, dtype='float')
anm_baseline *= anm_mag

time_series[anm_index:] = time_series[anm_index:] + anm_baseline

# Apply matrix profile
# mp = stumpy.stump(time_series, m=50, normalize=False)

import stumpy
def past_only_matrix_profile(T, m, normalize=False):
    n = len(T)
    profile = np.full(n - m + 1, 0)
    profile_idx = np.full(n - m + 1, -1)

    for i in range(m, n - m + 1):  # Start at m to ensure room for comparison
        Q = T[i:i+m]
        # D = stumpy.mass(Q, T[:i], normalize=normalize)
        D = stumpy.mass(Q, T[:int(0.6*len(T))], normalize=normalize)
        # D = stumpy.mass(Q, T, normalize=normalize)


        # Find best match in the past only
        min_idx = np.argmin(D)
        profile[i] = D[min_idx]
        profile_idx[i] = min_idx

    return profile, profile_idx

m = 52
mp, mpi = past_only_matrix_profile(time_series, m, normalize=False)
# Normalize mpi
# mp = (mp - np.min(mp)) / (np.max(mp) - np.min(mp))

ts_index = np.arange(len(time_series))

# Plot the time series and matrix profile
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 1, figsize=(10, 8))
axes[0].plot(ts_index, time_series, label='Time Series')

axes[1].plot(ts_index[:len(mp)], mp, label='Matrix Profile', color='orange')
plt.show()
