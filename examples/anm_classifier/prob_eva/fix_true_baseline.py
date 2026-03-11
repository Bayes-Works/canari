import ast
import numpy as np
import pandas as pd
from tqdm import tqdm


# Read csv saved_results/prob_eva/syn_simple_ts_results_rsic_v1_realjoint2_lttolt.csv
df_result = pd.read_csv("saved_results/prob_eva/rsic_bugged_results/syn_simple_ts_results_rsic_v1_wait3_lttolt_bug.csv")

df_result["true_LL_baseline"] = df_result["true_LL_baseline"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)
df_result["true_LT_baseline"] = df_result["true_LT_baseline"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

# Recalculate the true baselines
df = pd.read_csv("data/prob_eva_syn_time_series/syn_rsic_simple_ts_gen_lttolt.csv")

# Containers for restored data
restored_data = []
time_stamps = eval(df.iloc[0]["timestamp"], {"nan": float("nan")})
for _, row in df.iterrows():
    values = np.array(eval(row["values"], {"nan": float("nan")}), dtype=float)
    anomaly1_magnitude = float(row["anomaly1_magnitude"])
    anomaly2_magnitude = float(row["anomaly2_magnitude"])
    anomaly_start_index1 = int(row["anomaly_start_index1"])
    anomaly_start_index2 = int(row["anomaly_start_index2"])
    
    restored_data.append((values, anomaly1_magnitude, anomaly2_magnitude, anomaly_start_index1, anomaly_start_index2))

# True baselines
k_result_idx = 0
for m in range(10):
    for n in range(len(restored_data)//10):
        k = m + n * 10

        anm_mag1 = restored_data[k][1]
        anm_mag2 = restored_data[k][2]
        anm_start_index1 = restored_data[k][3]
        anm_start_index2 = restored_data[k][4]

        true_LL_baseline = np.zeros(len(df_result["true_LL_baseline"].iloc[k_result_idx]))
        true_LT_baseline = np.zeros(len(df_result["true_LT_baseline"].iloc[k_result_idx]))
        
        # # LL to LT anomaly
        # anm_mag2_perweek = anm_mag2 / 52
        # true_LL_baseline[anm_start_index1:] = anm_mag1
        # true_LL_baseline[anm_start_index2:] += np.arange(len(true_LL_baseline)-anm_start_index2) * anm_mag2_perweek
        # true_LT_baseline[anm_start_index2:] = anm_mag2_perweek

        # # LL to LL anomaly
        # true_LL_baseline[anm_start_index1:] = anm_mag1
        # true_LL_baseline[anm_start_index2:] += np.ones(len(true_LL_baseline)-anm_start_index2) * anm_mag2

        # # LT to LL anomaly
        # anm_mag1_perweek = anm_mag1 / 52
        # true_LL_baseline[anm_start_index1:] += np.arange(len(true_LL_baseline)-anm_start_index1) * anm_mag1_perweek
        # true_LL_baseline[anm_start_index2:] += anm_mag2
        # true_LT_baseline[anm_start_index1:] += anm_mag1_perweek

        # LT to LT anomaly
        anm_mag1_perweek = anm_mag1 / 52
        anm_mag2_perweek = anm_mag2 / 52
        true_LL_baseline[anm_start_index1:] += np.arange(len(true_LL_baseline)-anm_start_index1) * anm_mag1_perweek
        true_LL_baseline[anm_start_index2:] += np.arange(len(true_LL_baseline)-anm_start_index2) * anm_mag2_perweek
        true_LT_baseline[anm_start_index1:] += anm_mag1_perweek
        true_LT_baseline[anm_start_index2:] += anm_mag2_perweek

        # print("original true_LL_baseline: ", np.array(df_result["true_LL_baseline"].iloc[k_result_idx]))
        # print("fixed true_LL_baseline: ", true_LL_baseline)
        # print(true_LL_baseline == np.array(df_result["true_LL_baseline"].iloc[k_result_idx]))
        # print("anomaly1_magnitude: ", anm_mag1, "anomaly2_magnitude: ", anm_mag2)
        # print(anm_start_index1, anm_start_index2)
        # print('--------------------------------------------------------------------')

        # Replace the true baselines in df_result with the recalculated ones
        df_result.at[k_result_idx, "true_LL_baseline"] = str(true_LL_baseline.tolist())
        df_result.at[k_result_idx, "true_LT_baseline"] = str(true_LT_baseline.tolist())

        k_result_idx += 1

# Save the updated df_result to a new CSV
df_result.to_csv("saved_results/prob_eva/syn_simple_ts_results_rsic_v1_wait3_lttolt.csv", index=False)