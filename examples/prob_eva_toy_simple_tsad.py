import fire
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from src import (
    LocalLevel,
    LocalTrend,
    LocalAcceleration,
    LstmNetwork,
    Periodic,
    Autoregression,
    WhiteNoise,
    Model,
    plot_data,
    plot_prediction,
    plot_states,
)
from src.hsl_detection import hsl_detection
from examples import DataProcess
from pytagi import exponential_scheduler
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from matplotlib import gridspec
import pickle
import src.common as common
from src.data_visualization import add_dynamic_grids


# # Read data
data_file = "./data/toy_time_series/synthetic_simple_autoregression_periodic.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)

data_file_time = "./data/toy_time_series/synthetic_simple_autoregression_periodic_datetime.csv"
time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(time_series[0])
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values"]

# Data pre-processing
output_col = [0]
train_split=0.289
validation_split=0.0693*2

# Remove the last 52*5 rows in df_raw
train_split = train_split * len(df_raw) / len(df_raw[:-52*5])
validation_split = validation_split * len(df_raw) / len(df_raw[:-52*5])
df_raw = df_raw[:-52*5]

####################################################################
######################### Data generation model #########################
####################################################################
gen_model = Model(
    LocalTrend(mu_states=[0, 0], var_states=[1e-12, 1e-12], std_error=0),
    Periodic(period=52, mu_states=[0.5, 1], var_states=[1e-12, 1e-12]),
    Autoregression(std_error=0.05, phi=0.8, mu_states=[0], var_states=[0.08]),
)

data_processor = DataProcess(
    data=df_raw,
    time_covariates=["week_of_year"],
    train_split=train_split,
    validation_split=validation_split,
    output_col=output_col,
)

train_data, validation_data, test_data, normalized_data = data_processor.get_splits()
validation_start = df_raw.index[data_processor.validation_start]
test_start = df_raw.index[data_processor.test_start]
# validation_end = df_raw.index[data_processor.validation_end]


# Load model_dict from local
with open("saved_params/toy_simple_model.pkl", "rb") as f:
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

ltd_error = 1e-5

hsl_tsad_agent = hsl_detection(base_model=pretrained_model, data_processor=data_processor, drift_model_process_error_std=ltd_error)

# Get flexible drift model from the beginning
hsl_tsad_agent_pre = hsl_detection(base_model=pretrained_model.load_dict(pretrained_model.get_dict()), data_processor=data_processor, drift_model_process_error_std=ltd_error)
hsl_tsad_agent_pre.filter(train_data)
hsl_tsad_agent_pre.filter(validation_data)
hsl_tsad_agent.drift_model.var_states = hsl_tsad_agent_pre.drift_model.var_states


mu_obs_preds, std_obs_preds, mu_ar_preds, std_ar_preds = hsl_tsad_agent.filter(train_data, buffer_LTd=True)
mu_obs_preds, std_obs_preds, mu_ar_preds, std_ar_preds = hsl_tsad_agent.filter(validation_data, buffer_LTd=True)
# Run the generation model until validation data
gen_model.filter(train_data)
gen_model.filter(validation_data)

# hsl_tsad_agent.estimate_LTd_dist()
hsl_tsad_agent.mu_LTd = -1.035922643305238e-05
hsl_tsad_agent.LTd_std = 7.166353631122538e-05
hsl_tsad_agent.LTd_pdf = common.gaussian_pdf(mu = hsl_tsad_agent.mu_LTd, std = hsl_tsad_agent.LTd_std)

# hsl_tsad_agent.collect_synthetic_samples(num_time_series=100, save_to_path= 'data/hsl_tsad_training_samples/itv_learn_samples_toy_simple.csv')
hsl_tsad_agent.nn_train_with = 'tagiv'
hsl_tsad_agent.mean_train, hsl_tsad_agent.std_train, hsl_tsad_agent.mean_target, hsl_tsad_agent.std_target = 6.164014e-05, 0.00072832895, np.array([6.5932894e-04, 6.9455087e-02, 1.0722370e+02]), np.array([1.0831345e-02, 1.3456550e+00, 6.2564503e+01])
hsl_tsad_agent.learn_intervention(training_samples_path='data/hsl_tsad_training_samples/itv_learn_samples_toy_simple.csv', 
                                load_model_path='saved_params/NN_detection_model_toy_simple.pkl', max_training_epoch=50)
# hsl_tsad_agent.tune(decay_factor=0.95)
hsl_tsad_agent.LTd_pdf = common.gaussian_pdf(mu = hsl_tsad_agent.mu_LTd, std = hsl_tsad_agent.LTd_std * 0.6634204312890623)

gen_mode_copy = copy.deepcopy(gen_model)
# Store the states, mu_states, var_states, lstm_cell_states, and lstm_output_history of base_model
states_temp = copy.deepcopy(hsl_tsad_agent.base_model.states)
mu_states_temp = copy.deepcopy(hsl_tsad_agent.base_model.mu_states)
var_states_temp = copy.deepcopy(hsl_tsad_agent.base_model.var_states)
lstm_cell_states_temp = copy.deepcopy(hsl_tsad_agent.base_model.lstm_net.get_lstm_states())
lstm_output_history_temp = copy.deepcopy(hsl_tsad_agent.base_model.lstm_output_history)

drift_states_temp = copy.deepcopy(hsl_tsad_agent.drift_model.states)
mu_drift_states_temp = copy.deepcopy(hsl_tsad_agent.drift_model.mu_states)
var_drift_states_temp = copy.deepcopy(hsl_tsad_agent.drift_model.var_states)

current_time_step_temp = copy.deepcopy(hsl_tsad_agent.current_time_step)
p_anm_all_temp = copy.deepcopy(hsl_tsad_agent.p_anm_all)


num_test_ts = 5
anm_mag = 0.3/52
for k in range(num_test_ts):
    anm_start_index = np.random.randint(0, 52 * 2)
    num_time_steps = anm_start_index + 52 * 5

    anm_start_index_global = anm_start_index + len(df_raw) - len(test_data["y"])

    # Generate data from the current point
    gen_time_series, _, _, _ = gen_mode_copy.generate_time_series(num_time_series=1, num_time_steps=num_time_steps,
                                                                  add_anomaly=True, anomaly_mag_range = [anm_mag, anm_mag], anomaly_begin_range=[anm_start_index, anm_start_index+1])
    gen_time_series = gen_time_series[0]

    # Remove the last len(test_data["y"]) rows in df_raw
    df_raw = df_raw[:-len(test_data["y"])]

    # Genrate date_time from df_raw["date_time"][-1] with an interval of 7
    last_date_time = pd.to_datetime(test_start)
    date_time_array = np.array(pd.date_range(start=last_date_time, periods=len(gen_time_series), freq='7D'))

    new_df = pd.DataFrame({'values': gen_time_series}, index=pd.to_datetime(date_time_array))
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

    mu_obs_preds, std_obs_preds, mu_ar_preds, std_ar_preds = hsl_tsad_agent.detect(test_data, apply_intervention=True)

    if (np.array(hsl_tsad_agent.p_anm_all) > 0.5).any():
        anm_detected_index = np.where(np.array(hsl_tsad_agent.p_anm_all) > 0.5)[0][0]
    else:
        anm_detected_index = len(hsl_tsad_agent.p_anm_all)

    # Get true baseline
    norm_const_std = data_processor.norm_const_std[data_processor.output_col]
    anm_mag_normed = anm_mag / norm_const_std
    LL_baseline_true = np.zeros_like(df_raw)
    LT_baseline_true = np.zeros_like(df_raw)
    for i in range(1, len(df_raw)):
        if i > anm_start_index_global:
            LL_baseline_true[i] += anm_mag_normed * (i - anm_start_index_global)
            LT_baseline_true[i] += anm_mag_normed

    LL_baseline_true += model_dict['early_stop_init_mu_states'][0].item()
    LL_baseline_true = LL_baseline_true.flatten()
    LT_baseline_true = LT_baseline_true.flatten()

    # Compute MSE for SKF baselines
    mu_LL_states = hsl_tsad_agent.base_model.states.get_mean(states_type='prior', states_name=["local level"])["local level"]
    mu_LT_states = hsl_tsad_agent.base_model.states.get_mean(states_type='prior', states_name=["local trend"])["local trend"]
    mse_LL = metric.mse(
        mu_LL_states[anm_start_index_global:],
        LL_baseline_true[anm_start_index_global:],
    )
    mse_LT = metric.mse(
        mu_LT_states[anm_start_index_global:],
        LT_baseline_true[anm_start_index_global:],
    )

    #  Plot
    state_type = "prior"
    #  Plot states from pretrained model
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(5, 1)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])
    ax3 = plt.subplot(gs[3])
    ax4 = plt.subplot(gs[4])
    from src.data_visualization import determine_time
    time = determine_time(data_processor, len(normalized_data["y"]))
    plot_data(
        data_processor=data_processor,
        normalization=True,
        plot_column=output_col,
        validation_label="y",
        sub_plot=ax0,
    )
    plot_states(
        data_processor=data_processor,
        normalization=True,
        # states=pretrained_model.states,
        states=hsl_tsad_agent.base_model.states,
        states_type=state_type,
        states_to_plot=['local level'],
        sub_plot=ax0,
    )
    ax0.set_xticklabels([])
    ax0.plot(time, LL_baseline_true, color='k', linestyle='--')
    ax0.axvline(x=time[anm_start_index_global], color='r', linestyle='--')
    ax0.set_title(f"IL, mse_LL = {mse_LL:.3e}, mse_LT = {mse_LT:.3e}, detection_time = {anm_detected_index - anm_start_index_global}")
    plot_states(
        data_processor=data_processor,
        normalization=True,
        states=hsl_tsad_agent.base_model.states,
        states_type=state_type,
        states_to_plot=['local trend'],
        sub_plot=ax1,
    )
    ax1.set_xticklabels([])
    ax1.plot(time, LT_baseline_true, color='k', linestyle='--')
    plot_states(
        data_processor=data_processor,
        normalization=True,
        states=hsl_tsad_agent.base_model.states,
        states_type=state_type,
        states_to_plot=['lstm'],
        sub_plot=ax2,
    )
    ax2.set_xticklabels([])
    plot_states(
        data_processor=data_processor,
        normalization=True,
        states=hsl_tsad_agent.base_model.states,
        states_type=state_type,
        states_to_plot=['autoregression'],
        sub_plot=ax3,
    )
    ax3.set_xticklabels([])

    ax4.plot(time, hsl_tsad_agent.p_anm_all, color='b')
    ax4.set_ylabel("p_anm")
    ax4.set_xlim(ax0.get_xlim())
    ax4.set_ylim(-0.05, 1.05)
    add_dynamic_grids(ax4, time)
    plt.show()

    # Put back the states, mu_states, var_states, lstm_cell_states, and lstm_output_history of base_model
    hsl_tsad_agent.base_model.states = copy.deepcopy(states_temp)
    hsl_tsad_agent.base_model.mu_states = copy.deepcopy(mu_states_temp)
    hsl_tsad_agent.base_model.var_states = copy.deepcopy(var_states_temp)
    hsl_tsad_agent.base_model.lstm_net.set_lstm_states(lstm_cell_states_temp)
    hsl_tsad_agent.base_model.lstm_output_history = copy.deepcopy(lstm_output_history_temp)
    hsl_tsad_agent.drift_model.states = copy.deepcopy(drift_states_temp)
    hsl_tsad_agent.drift_model.mu_states = copy.deepcopy(mu_drift_states_temp)
    hsl_tsad_agent.drift_model.var_states = copy.deepcopy(var_drift_states_temp)
    hsl_tsad_agent.current_time_step = copy.deepcopy(current_time_step_temp)
    hsl_tsad_agent.p_anm_all = copy.deepcopy(p_anm_all_temp)

