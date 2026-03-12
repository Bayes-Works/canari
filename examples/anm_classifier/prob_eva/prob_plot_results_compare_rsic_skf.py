# Read CSV file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import ast

from matplotlib import ticker
from examples.anm_classifier.prob_eva.prob_process_csv_results import _process_detection_df
from examples.anm_classifier.prob_eva.prob_process_csv_results_bl import _process_detection_df_bl
from examples.anm_classifier.prob_eva.prob_process_csv_results_compare_map import _process_detection_compare_map

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 
params = {'text.usetex' : True,
          'font.size' : 12,
          'font.family' : 'lmodern',
          'lines.linewidth' : 1,
          }
plt.rcParams.update(params)
# plt.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'

# Get the total length of the test time series
test_ts_df = pd.read_csv("data/prob_eva_syn_time_series/syn_rsic_simple_ts_gen_lltolt.csv")
test_ts_len = len(np.array(eval(test_ts_df.iloc[0]["values"])).flatten())

_process_detection_compare_map(
    test_ts_len=test_ts_len,
    csv_rsic_path="saved_results/prob_eva/syn_simple_ts_results_rsic_v1_realjoint2_lltolt.csv",
    csv_skf_path="saved_results/prob_eva/syn_simple_ts_results_skf_lltolt.csv",
    plot_detection_map = True,
)