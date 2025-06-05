import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from canari.component import LocalTrend, LstmNetwork, Periodic, Autoregression
from canari import (
    DataProcess,
    Model,
    plot_data,
    plot_prediction,
    plot_states,
)
import pickle
from tqdm import tqdm
from matplotlib import gridspec
from pytagi import Normalizer as normalizer


# # Read data
data_file = "./data/benchmark_data/test_5_data.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(df_raw.iloc[:, 0])
df_raw = df_raw.iloc[:, 1:]
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values", "water_level", "temp_min", "temp_max"]
df_raw = df_raw.iloc[:, :-3]

# Data pre-processing
output_col = [0]
train_split=0.289
validation_split=0.0693*2

time_covariates = ["week_of_year"]
data_processor = DataProcess(
    data=df_raw,
    time_covariates=time_covariates,
    train_split=train_split,
    validation_split=validation_split,
    output_col=output_col,
)

train_data, validation_data, test_data, normalized_data = data_processor.get_splits()
train_index, val_index, test_index = data_processor.get_split_indices()
validation_start = df_raw.index[data_processor.validation_start]
test_start = df_raw.index[data_processor.test_start]
time_covariate_info = {'initial_time_covariate': data_processor.data.values[val_index[-1], data_processor.covariates_col].item(),
                                'mu': data_processor.std_const_mean[data_processor.covariates_col], 
                                'std': data_processor.std_const_std[data_processor.covariates_col]}

# Load model_dict from local
with open("saved_params/real_ts5_model_rebased.pkl", "rb") as f:
    model_dict = pickle.load(f)

####################################################################
######################### Pretrained model #########################
####################################################################
LSTM = LstmNetwork(
        look_back_len=19,
        num_features=2,
        num_layer=1,
        num_hidden_unit=50,
        device="cpu",
    )

print("phi_AR =", model_dict['states_optimal'].mu_prior[-1][model_dict['phi_index']].item())
print("sigma_AR =", np.sqrt(model_dict['states_optimal'].mu_prior[-1][model_dict['W2bar_index']].item()))


pretrained_model = Model(
    LocalTrend(mu_states=model_dict['early_stop_init_mu_states'][0:2].reshape(-1), var_states=[1e-12, 1e-12]),
    LSTM,
    Autoregression(std_error=np.sqrt(model_dict['states_optimal'].mu_prior[-1][model_dict['W2bar_index']].item()), 
                   phi=model_dict['states_optimal'].mu_prior[-1][model_dict['phi_index']].item(), 
                   mu_states=[model_dict['early_stop_init_mu_states'][model_dict['autoregression_index']].item()], 
                   var_states=[model_dict['early_stop_init_var_states'][model_dict['autoregression_index'], model_dict['autoregression_index']].item()]),
)

pretrained_model.lstm_net.load_state_dict(model_dict["lstm_network_params"])

pretrained_model.filter(train_data)
pretrained_model.filter(validation_data)

gen_mode_copy = copy.deepcopy(pretrained_model)
# Get LSTM initializations
if "lstm" in pretrained_model.states_name:
    gen_mode_copy.lstm_net = pretrained_model.lstm_net
    if (
        pretrained_model.lstm_output_history.mu is not None
        and pretrained_model.lstm_output_history.var is not None
    ):
        lstm_output_history_mu_temp = copy.deepcopy(pretrained_model.lstm_output_history.mu)
        lstm_output_history_var_temp = copy.deepcopy(
            pretrained_model.lstm_output_history.var
        )
        lstm_output_history_exist = True
    else:
        lstm_output_history_exist = False

    lstm_cell_states = pretrained_model.lstm_net.get_lstm_states()


num_test_ts = 10
time_series_all = []
anm_mag_all = np.concatenate([np.arange(0.01, 0.11, 0.01), np.arange(0.2, 1.01, 0.1)])/52
# anm_mag_all = [0, 0.1]

for i, anm_mag in tqdm(enumerate(anm_mag_all)):
    for k in range(num_test_ts):
        if "lstm" in pretrained_model.states_name:
            # Reset lstm cell states
            gen_mode_copy.lstm_net.set_lstm_states(lstm_cell_states)
            # Reset lstm output history
            if lstm_output_history_exist:
                gen_mode_copy.lstm_output_history.mu = copy.deepcopy(
                    lstm_output_history_mu_temp
                )
                gen_mode_copy.lstm_output_history.var = copy.deepcopy(
                    lstm_output_history_var_temp
                )
            else:
                gen_mode_copy.lstm_output_history.initialize(
                    gen_mode_copy.lstm_net.lstm_look_back_len
                )

        anm_start_index = np.random.randint(0, 52 * 2)
        num_time_steps = anm_start_index + 52 * 5

        anm_start_index_global = anm_start_index + len(df_raw) - len(test_data["y"])

        # Generate data from the current point
        gen_time_series, _, _, _ = gen_mode_copy.generate_time_series(num_time_series=1, num_time_steps=num_time_steps, time_covariates=time_covariates, time_covariate_info=time_covariate_info,
                                                                    add_anomaly=True, anomaly_mag_range = [anm_mag, anm_mag], anomaly_begin_range=[anm_start_index, anm_start_index+1])
        gen_time_series = normalizer.unstandardize(
            gen_time_series,
            data_processor.std_const_mean[output_col],
            data_processor.std_const_std[output_col],
        )
        gen_time_series = gen_time_series[0]

        # Remove the last len(test_data["y"]) rows in df_raw
        df_raw = df_raw[:-len(test_data["y"])]

        # Genrate date_time from df_raw["date_time"][-1] with an interval of 7
        last_date_time = pd.to_datetime(test_start)
        date_time_array = np.array(pd.date_range(start=last_date_time, periods=len(gen_time_series), freq='7D'))

        new_df = pd.DataFrame({'values': gen_time_series}, index=pd.to_datetime(date_time_array))

        # Convert timestamps and values to string representations
        timestamps_str = str(list(pd.to_datetime(date_time_array).strftime('%Y-%m-%d %H:%M:%S')))
        values_str = str(list(gen_time_series))

        time_series_all.append([timestamps_str, values_str, anm_mag, anm_start_index])

        df_raw = pd.concat([df_raw, new_df])

        # df_raw["values"].iloc[-len(test_data["y"]):] = gen_time_series
        data_processor = DataProcess(
            data=df_raw,
            time_covariates=["week_of_year"],
            validation_start = validation_start,
            test_start = test_start,
            output_col=output_col,
        )
        train_data, validation_data, test_data, normalized_data = data_processor.get_splits()

        # #  Plot
        # state_type = "prior"
        # #  Plot states from pretrained model
        # fig = plt.figure(figsize=(10, 2))
        # gs = gridspec.GridSpec(1, 1)
        # ax0 = plt.subplot(gs[0])
        # time = data_processor.get_time(split="all")
        # plot_data(
        #     data_processor=data_processor,
        #     standardization=True,
        #     plot_column=output_col,
        #     validation_label="y",
        #     sub_plot=ax0,
        # )
        # ax0.axvline(x=time[anm_start_index_global], color='r', linestyle='--')
        # plt.show()

# # Save to CSV
# df_time_series_all = pd.DataFrame(time_series_all, columns=["timestamps", "values", "anomaly_magnitude", "anomaly_start_index"])
# df_time_series_all.to_csv("data/prob_eva_syn_time_series/real_ts5_tsgen.csv", index=False)