# # Run tourism monthly 
import pandas as pd
import numpy as np
from bm.tourism_month import tourism_month
import pickle
import time
from tqdm import tqdm

time_start = time.time()

# Read data
data_train_file = "./data/tourism/monthly_in.csv"
df_train = pd.read_csv(data_train_file, skiprows=0, delimiter=",", header=None)
data_test_file = "./data/tourism/monthly_oos.csv"
df_test = pd.read_csv(data_test_file, skiprows=0, delimiter=",", header=None)

time_series = np.arange(366)
saved_result = {
    "states": {},
    "mu_test": {},
    "var_test": {},
}

for ts in tqdm(time_series, desc="Time series"):
    mu_test, var_test, states = tourism_month(df_train, df_test, ts)
    saved_result["mu_test"][ts] = mu_test
    saved_result["var_test"][ts] = var_test
    saved_result["states"][ts] = states

with open("bm/results/tourism_month.pkl", "wb") as f:
    pickle.dump(saved_result, f)

time_end = time.time()
print(f"Runtime: {time_end - time_start:.2f} seconds")
