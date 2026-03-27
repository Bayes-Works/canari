import fire
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from canari import (
    DataProcess,
    Model,
    plot_states,
    plot_data, 
    plot_prediction,
)
from canari.component import LocalTrend, KernelRegression, WhiteNoise
import json

# Read data
with open("examples/benchmark/BM_metadata.json", "r") as f:
    metadata = json.load(f)

benchmark_no: str = ["4"]
for benchmark in benchmark_no:
    config = metadata[benchmark]

    ######### Data processing #########
    # Read data
    data_file = config["data_path"]
    df = pd.read_csv(data_file, skiprows=0, delimiter=",")
    date_time = pd.to_datetime(df["date"])
    df = df.drop("date", axis=1)
    df.index = date_time
    df.index.name = "date_time"
    # Data pre-processing
    df = DataProcess.add_lagged_columns(df, config["lag_vector"])
    output_col = config["output_col"]
    data_processor = DataProcess(
        data=df,
        # time_covariates=config["time_covariates"],
        train_split=0.8,
        validation_split=0.2,
        output_col=output_col,
    )
    train_data, validation_data, _, all_data = data_processor.get_splits()

# Components
model = Model(
    LocalTrend(),
    KernelRegression(period=52, 
                    kernel_length=0.8, 
                    num_control_point=10,
                    mu_control_point=0.1,
                    var_control_point=0.1,
                    ),
    WhiteNoise(std_error=1e-1)
)
model.auto_initialize_baseline_states(train_data["y"][0:52*3])

mu_obs_preds,std_obs_preds,_ = model.filter(data=train_data)
model.smoother()

#  Plot
fig, axes=plot_states(
    data_processor=data_processor,
    states=model.states,
    states_to_plot=["level", "trend", "kernel regression", "white noise"],
    standardization=False,
    states_type="smooth",
)
plot_data(
    data_processor=data_processor,
    standardization=False,
    plot_column=output_col,
    sub_plot=axes[0],
)
plt.show()
