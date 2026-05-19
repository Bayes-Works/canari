from merlion.utils import TimeSeries
from ts_datasets.anomaly import MSL
import matplotlib.pyplot as plt

# time_series, metadata = MSL()[0]
# train_data = TimeSeries.from_pd(time_series[metadata.trainval])
# test_data = TimeSeries.from_pd(time_series[~metadata.trainval])
# test_labels = TimeSeries.from_pd(metadata.anomaly[~metadata.trainval])

# Read local csv
import pandas as pd
import numpy as np
from tqdm import tqdm
import copy

# df = pd.read_csv("/Users/zhanwenxin/code/Merlion/data/hq/test_1_data.csv")
# df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
# df = df.set_index(df.columns[0])


# data_file = "/Users/zhanwenxin/code/Merlion/data/hq/test_2_data.csv"
# df = pd.read_csv(data_file, skiprows=1, delimiter=";", header=None)
# time_series = pd.to_datetime(df.iloc[:, 4])
# df = df.iloc[:, 6].to_frame()
# df.index = time_series
# df.index.name = "date_time"
# df.columns = ["value"]
# df = df.resample("W").mean()
# df = df.iloc[30:, :]

# data_file = "/Users/zhanwenxin/code/Merlion/data/hq/test_11_data.csv"
# df = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
# time_series = pd.to_datetime(df.iloc[:, 0])
# df = df.iloc[:, 1:]
# df.index = time_series
# df.index.name = "date_time"
# df.columns = ["value"]

# data_file = "/Users/zhanwenxin/code/Merlion/data/hq/test_4_data.csv"
# df = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
# time_series = pd.to_datetime(df.iloc[:, 0])
# df = df.iloc[:, 1:]
# df.index = time_series
# df.index.name = "date_time"
# df.columns = ["value", "water_level", "temp_min", "temp_max"]
# df = df.iloc[:, :-3]

# data_file = "/Users/zhanwenxin/code/Merlion/data/hq/test_5_data.csv"
# df = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
# time_series = pd.to_datetime(df.iloc[:, 0])
# df = df.iloc[:, 1:]
# df.index = time_series
# df.index.name = "date_time"
# df.columns = ["value", "water_level", "temp_min", "temp_max"]
# df = df.iloc[:, :-3]

# data_file = "/Users/zhanwenxin/code/Merlion/data/hq/test_6_data.csv"
# df = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
# time_series = pd.to_datetime(df.iloc[:, 0])
# df = df.iloc[:, 1:]
# # Remove the first 52 rows
# df.index = time_series
# df.index.name = "date_time"
# df.columns = ["value", "water_level", "temp_min", "temp_max"]
# df = df.iloc[:, :-3]
# df = df.iloc[52:, :]

# data_file = "/Users/zhanwenxin/code/Merlion/data/hq/test_7_data.csv"
# df = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
# time_series = pd.to_datetime(df.iloc[:, 0])
# df = df.iloc[:, 1:]
# df.index = time_series
# df.index.name = "date_time"
# df.columns = ["value", "water_level", "temp_min", "temp_max"]
# df = df.iloc[:, :-3]

# data_file = "/Users/zhanwenxin/code/Merlion/data/hq/test_8_data.csv"
# df = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
# time_series = pd.to_datetime(df.iloc[:, 0])
# df = df.iloc[:, 1:]
# df.index = time_series
# df.index.name = "date_time"
# df.columns = ["value", "water_level", "temp_min", "temp_max"]
# df = df.iloc[:, :-3]

# data_file = "/Users/zhanwenxin/code/Merlion/data/hq/test_9_data.csv"
# df = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
# time_series = pd.to_datetime(df.iloc[:, 0])
# df = df.iloc[:, 1:]
# df.index = time_series
# df.index.name = "date_time"
# df.columns = ["value", "water_level", "temp_min", "temp_max"]
# df = df.iloc[:, :-3]

data_file = "/Users/zhanwenxin/code/Merlion/data/hq/test_10_data.csv"
df = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(df.iloc[:, 0])
df = df.iloc[:, 1:]
df.index = time_series
df.index.name = "date_time"
df.columns = ["value", "water_level", "temp_min", "temp_max"]
df = df.iloc[:, :-3]

df.index = df.index - df.index[0] + pd.Timestamp("1700-01-01")
df.index = pd.date_range(start=df.index[0], periods=len(df), freq='D')


# Take the first 30% as training data in the training dataset
train_end = int(len(df) * 0.33375)
train_df = df.iloc[:train_end, :]

mean_train = train_df.mean()
std_train = train_df.std()

print(type(std_train))
print(std_train)

# Create the train_labels pd.dataframe for train_df full of zeros
train_labels = pd.DataFrame(0, index=train_df.index, columns=["anomaly"])

# Extend time series and add anomaly
train_anm_df_origin = train_df.copy()
train_anm_labels_origin = train_labels.copy()
for i in tqdm(range(50)):
    train_anm_df = train_anm_df_origin.copy()
    train_anm_labels = train_anm_labels_origin.copy()
    # Denormalize the anomaly magnitude range
    anm_mag_range = np.array([-1/52, 1/52]) * std_train.value
    anm_mag = np.random.uniform(anm_mag_range[0], anm_mag_range[1])
    # 50% of the time series will have anomalies
    anm_mag = anm_mag if np.random.rand() > 0.5 else 0
    time_anomaly = np.random.randint(int(len(train_anm_df_origin)/4), int(len(train_anm_df_origin)*3/8))
    anm_baseline = np.arange(len(train_anm_df_origin)) * anm_mag
    anm_baseline[time_anomaly:] -= anm_baseline[time_anomaly]
    anm_baseline[:time_anomaly] = 0
    train_anm_df = train_anm_df.add(anm_baseline, axis=0)
    if anm_mag != 0:
      train_anm_labels.iloc[time_anomaly:, 0] = 1

    # Start the train_anm_df index from the last timestamp index of train_df + 1 time unit
    last_timestamp = train_df.index[-1]
    time_unit = train_df.index[1] - train_df.index[0]
    new_index = [last_timestamp + time_unit * (i + 1) for i in range(len(train_anm_df))]
    train_anm_df.index = new_index
    train_anm_labels.index = new_index

    # Concatenate train_anm_df to train_df
    train_df = pd.concat([train_df, train_anm_df])
    train_labels = pd.concat([train_labels, train_anm_labels])

# Plot train_df and train_labels
plt.figure(figsize=(15, 5))
plt.plot(train_df.index, train_df.values, label='Training Data with Anomalies')
plt.scatter(train_labels[train_labels['anomaly'] == 1].index,
            train_df.loc[train_labels['anomaly'] == 1].values,
            color='red', label='Anomalies')
plt.title('Training Data with Injected Anomalies')
plt.show()

train_data = TimeSeries.from_pd(train_df)
train_labels = TimeSeries.from_pd(train_labels)


# val_data = TimeSeries.from_pd(df.iloc[train_end:val_end, :])
# test_data = TimeSeries.from_pd(df.iloc[train_end:, :])
test_data = TimeSeries.from_pd(df)

# Create the test labels (here we just use all zeros as an example)
test_labels = TimeSeries.from_pd(pd.DataFrame(0, index=df.iloc[train_end:, :].index, columns=["anomaly"]))


# from merlion.models.defaults import DefaultDetectorConfig, DefaultDetector
# model = DefaultDetector(DefaultDetectorConfig())

from merlion.models.anomaly.lstm_ed import LSTMEDConfig, LSTMED
model = LSTMED(LSTMEDConfig(num_epochs=100, sequence_len=52))

# from merlion.models.anomaly.autoencoder import AutoEncoderConfig, AutoEncoder
# model = AutoEncoder(AutoEncoderConfig())
model.train(train_data=train_data, anomaly_labels=train_labels)
# model.train(train_data=train_data, train_config=None, anomaly_labels=train_labels)
test_pred = model.get_anomaly_score(time_series=test_data)

scores = model.get_anomaly_label(time_series=test_data)
scores = scores.univariates[scores.names[0]]

# Set score threshold to 1.1 times the maximum score in the training data
scores_threshold = max(1.1 * scores[:train_end].max(), 0.1)
print(scores_threshold)
1/0

pred_anm = pd.DataFrame(0, index=df.iloc[train_end:, :].index, columns=["anomaly"])
pred_anm[scores[train_end:] > scores_threshold] = 1
pred_anm = TimeSeries.from_pd(pred_anm)

from merlion.plot import plot_anoms
import matplotlib.pyplot as plt
fig, ax = model.plot_anomaly(time_series=test_data)
plot_anoms(ax=ax, anomaly_labels=pred_anm)
plt.show()

# from merlion.evaluate.anomaly import TSADMetric
# p = TSADMetric.Precision.value(ground_truth=test_labels, predict=test_pred)
# r = TSADMetric.Recall.value(ground_truth=test_labels, predict=test_pred)
# f1 = TSADMetric.F1.value(ground_truth=test_labels, predict=test_pred)
# mttd = TSADMetric.MeanTimeToDetect.value(ground_truth=test_labels, predict=test_pred)
# print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}\n"
#       f"Mean Time To Detect: {mttd}")