# # Run tourism monthly 
import pandas as pd
import numpy as np
from bm.tourism_month_filter import tourism_month
import pickle
import time
from tqdm import tqdm
from bm.utils import p50, p90


with open("saved_results/bm/tourism_quarter_6.pkl", "rb") as f:
    results = pickle.load(f)

p50 = results["p50"]
p90 = results["p90"]
print(f"p50: {p50:.4f}")
print(f"p90: {p90:.4f}")
