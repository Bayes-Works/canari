import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec


samples = pd.read_csv('data/hsl_tsad_training_samples/hsl_tsad_train_samples_simpleTS_fourrier_300_fix_anm_mag_small_ltd_error.csv')
samples['LTd_history'] = samples['LTd_history'].apply(lambda x: list(map(float, x[1:-1].split(','))))
# Convert samples['anm_develop_time'] to float
samples['anm_develop_time'] = samples['anm_develop_time'].apply(lambda x: float(x))

for i in range(len(samples['LTd_history'])):
    if samples['anm_develop_time'][i] == 0:
        print('------------------ New time series ------------------')
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(1, 1)
    ax0 = plt.subplot(gs[0])
    ax0.plot(samples['LTd_history'][i])
    ax0.axhline(y=0, color='r', linestyle='--')
    plt.show()