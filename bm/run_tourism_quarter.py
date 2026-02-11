# # Run tourism monthly 
import pandas as pd
import numpy as np
from bm.tourism_quarter import tourism_quarter
import pickle
import time
from tqdm import tqdm
from bm.utils import p50, p90

time_start = time.time()

# Read data
data_train_file = "./data/tourism/quarterly_in.csv"
df_train = pd.read_csv(data_train_file, skiprows=0, delimiter=",", header=None)
data_test_file = "./data/tourism/quarterly_oos.csv"
df_test = pd.read_csv(data_test_file, skiprows=0, delimiter=",", header=None)

time_series = np.arange(427)
mu_test_all = np.zeros((8,len(time_series)))
std_test_all = np.zeros((8,len(time_series)))
test_obs_all = np.zeros((8,len(time_series)))
saved_result = {
    "states": {},
    "mu_test": {},
    "std_test": {},
    "test_obs": {},
    "p50": {},
    "p90": {},
}

for ts in tqdm(time_series, desc="Time series"):
    mu_test, std_test, states, test_obs = tourism_quarter(df_train, df_test, ts)
    saved_result["states"][ts] = states
    mu_test_all[:,ts] = mu_test.flatten()
    std_test_all[:,ts] = std_test.flatten()
    test_obs_all[:,ts] = test_obs.flatten()

# Metrics
p50_overall = p50(test_obs_all, mu_test_all, std_test_all)
p90_overall = p90(test_obs_all, mu_test_all, std_test_all)

saved_result["mu_test"] = mu_test_all
saved_result["std_test"] = std_test_all
saved_result["p50"] = p50_overall
saved_result["p90"] = p90_overall

with open("bm/results/tourism_quarter.pkl", "wb") as f:
    pickle.dump(saved_result, f)

time_end = time.time()
print(f"Runtime: {time_end - time_start:.2f} seconds")


print(f"p50: {p50_overall:.4f}")
print(f"p90: {p90_overall:.4f}")
