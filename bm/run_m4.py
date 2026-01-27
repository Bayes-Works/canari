# # Run tourism monthly 
import pandas as pd
import numpy as np
from bm.m4 import m4_hour
import pickle
import time
from tqdm import tqdm

time_start = time.time()

# Read data
data_train_file = "./data/m4/Hourly-train.csv"
df_train = pd.read_csv(data_train_file, skiprows=1, delimiter=",", header=None)
data_test_file = "./data/m4/Hourly-test.csv"
df_test = pd.read_csv(data_test_file, skiprows=1, delimiter=",", header=None)

time_series = np.arange(414)
saved_result = {
    "states": {},
    "mu_test": {},
    "var_test": {},
}

for ts in tqdm(time_series, desc="Time series"):
    mu_test, var_test, states = m4_hour(df_train, df_test, ts)
    saved_result["mu_test"][ts] = mu_test
    saved_result["var_test"][ts] = var_test
    saved_result["states"][ts] = states

with open("bm/results/tourism_month.pkl", "wb") as f:
    pickle.dump(saved_result, f)

time_end = time.time()
print(f"Runtime: {time_end - time_start:.2f} seconds")
