import pandas as pd
import numpy as np
import ast

# Load the CSV
df = pd.read_csv("data/prob_eva_syn_time_series/real_ts5_tsgen.csv")
# Containers for restored data
restored_data = []
for _, row in df.iterrows():
    # Convert string to list, then to desired type
    timestamps = pd.to_datetime(ast.literal_eval(row["timestamps"]))
    values = np.array(ast.literal_eval(row["values"]), dtype=float)
    anomaly_magnitude = float(row["anomaly_magnitude"])
    anomaly_start_index = int(row["anomaly_start_index"])
    
    restored_data.append((timestamps, values, anomaly_magnitude, anomaly_start_index))

print(len(restored_data[2][1]), restored_data[2][3]+52*5)
