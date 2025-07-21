import fire
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pytagi import metric
from pytagi import Normalizer
from canari import (
    DataProcess,
    Model,
    ModelOptimizer,
    SKF,
    SKFOptimizer,
    plot_data,
    plot_prediction,
    plot_skf_states,
    plot_states,
)
from canari.component import LocalTrend, LocalAcceleration, LstmNetwork, WhiteNoise, Autoregression
import pickle

SKF_std_transition_error_fix = None
SKF_norm_to_abnorm_prob_fix = None


def main(
    num_trial_optimization: int = 50,
    param_optimization: bool = True,
):
    # Read data
    data_file = "./data/benchmark_data/test_9_data.csv"
    df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
    time_series = pd.to_datetime(df_raw.iloc[:, 0])
    df_raw = df_raw.iloc[:, 1:]
    df_raw.index = time_series
    df_raw.index.name = "date_time"
    df_raw.columns = ["values", "water_level", "temp_min", "temp_max"]
    # df_raw = df_raw.iloc[:, :-3]
    lags = [0, 4, 4, 4]
    df_raw = DataProcess.add_lagged_columns(df_raw, lags)

    # Data pre-processing
    output_col = [0]
    data_processor = DataProcess(
        data=df_raw,
        time_covariates=["week_of_year"],
        train_split=0.35354,
        validation_split=0.087542,
        output_col=output_col,
    )
    data_processor.scale_const_mean, data_processor.scale_const_std = Normalizer.compute_mean_std(
                data_processor.data.iloc[0 : 52].values
            )
    train_data, validation_data, test_data, all_data = data_processor.get_splits()

    ########################################
    ########################################

    # Load model_dict from local
    with open("saved_params/real_ts9_tsmodel_raw_expl.pkl", "rb") as f:
        model_dict = pickle.load(f)

    LSTM = LstmNetwork(
            look_back_len=33,
            num_features=17,
            num_layer=1,
            num_hidden_unit=50,
            device="cpu",
        )
    
    # phi_index = model_dict["states_name"].index("phi")
    # W2bar_index = model_dict["states_name"].index("W2bar")
    # autoregression_index = model_dict["states_name"].index("autoregression")

    # print("phi_AR =", model_dict['states_optimal'].mu_prior[-1][phi_index].item())
    # print("sigma_AR =", np.sqrt(model_dict['states_optimal'].mu_prior[-1][W2bar_index].item()))

    def initialize_model(param, train_data, validation_data):
        model = Model(
            # LocalTrend(mu_states=model_dict['states_optimal'].mu_prior[0][0:2].reshape(-1), var_states=[1e-12, 1e-12]),
            LocalTrend(mu_states=model_dict['states_optimal'].mu_prior[0][0:2].reshape(-1), var_states=[model_dict['states_optimal'].var_prior[0][0,0].item(), model_dict['states_optimal'].var_prior[0][1,1].item()]),
            LSTM,
            # Autoregression(std_error=np.sqrt(model_dict['states_optimal'].mu_prior[-1][W2bar_index].item()), 
            #             # phi=model_dict['states_optimal'].mu_prior[-1][phi_index].item(),
            #             phi=0,
            #             mu_states=[model_dict["mu_states"][autoregression_index].item()], 
            #             var_states=[model_dict["var_states"][autoregression_index, autoregression_index].item()]),
            WhiteNoise(std_error=0.02222769157019269),
        )

        model.lstm_net.load_state_dict(model_dict["lstm_network_params"])

        return model

    param = {}

    model_optim = initialize_model(param, train_data, validation_data)

    # Save best model for SKF analysis later
    model_optim_dict = model_optim.get_dict()

    ########################################
    ########################################

    # Parameter optimization for SKF
    def initialize_skf(skf_param_space, model_param: dict):
        norm_model = Model.load_dict(model_param)
        abnorm_model = Model(
            # LocalAcceleration(mu_states=[model_dict["mu_states"][0].item(), model_dict["mu_states"][1].item(), 0], var_states=[1e-12, 1e-12, 1e-4]),
            LocalAcceleration(),
            LstmNetwork(),
            # Autoregression(std_error=np.sqrt(model_dict['states_optimal'].mu_prior[-1][W2bar_index].item()), 
            #             # phi=model_dict['states_optimal'].mu_prior[-1][phi_index].item(), 
            #             phi=0,
            #             mu_states=[model_dict["mu_states"][autoregression_index].item()], 
            #             var_states=[model_dict["var_states"][autoregression_index, autoregression_index].item()]),
            WhiteNoise(),
        )
        norm_model.lstm_net.load_state_dict(model_dict["lstm_network_params"])
        abnorm_model.lstm_net.load_state_dict(model_dict["lstm_network_params"])

        skf = SKF(
            norm_model=norm_model,
            abnorm_model=abnorm_model,
            std_transition_error=skf_param_space["std_transition_error"],
            norm_to_abnorm_prob=skf_param_space["norm_to_abnorm_prob"],
        )
        skf.save_initial_states()
        return skf

    # Define parameter search space
    slope_upper_bound = 5e-2
    slope_lower_bound = 1e-3

    # Plot synthetic anomaly
    synthetic_anomaly_data = DataProcess.add_synthetic_anomaly(
        train_data,
        num_samples=1,
        slope=[slope_lower_bound, slope_upper_bound],
    )
    plot_data(
        data_processor=data_processor,
        standardization=True,
        plot_validation_data=False,
        plot_test_data=False,
        plot_column=output_col,
    )
    train_time = data_processor.get_time("train")
    for ts in synthetic_anomaly_data:
        plt.plot(train_time, ts["y"])
    plt.legend(
        [
            "data without anomaly",
            "",
            "smallest anomaly tested",
            "largest anomaly tested",
        ]
    )
    plt.title("Train data with added synthetic anomalies")
    plt.show()

    if param_optimization:
        skf_param_space = {
            "std_transition_error": [1e-6, 1e-3],
            "norm_to_abnorm_prob": [1e-6, 1e-3],
            "slope": [slope_lower_bound, slope_upper_bound],
            "threshold_anm_prob": [1e-2, 1.],
        }
        # Define optimizer
        skf_optimizer = SKFOptimizer(
            initialize_skf=initialize_skf,
            model_param=model_optim_dict,
            param_space=skf_param_space,
            data=train_data,
            num_synthetic_anomaly=50,
            num_optimization_trial=num_trial_optimization * 2,
        )
        skf_optimizer.optimize()

        # Get parameters
        skf_param = skf_optimizer.get_best_param()
    else:
        skf_param = {
            "std_transition_error": SKF_std_transition_error_fix,
            "norm_to_abnorm_prob": SKF_norm_to_abnorm_prob_fix,
        }

    print("SKF model parameters used:", skf_param)
    skf_optim = initialize_skf(skf_param, model_optim_dict)

    # Detect anomaly
    filter_marginal_abnorm_prob, states = skf_optim.filter(data=all_data)
    # smooth_marginal_abnorm_prob, states = skf_optim.smoother()

    # Plotting SKF results
    fig, ax = plot_skf_states(
        data_processor=data_processor,
        states=states,
        states_type="prior",
        # states_to_plot=["level", "trend", "lstm", "autoregression"],
        model_prob=filter_marginal_abnorm_prob,
        standardization=False,
    )
    fig.suptitle("SKF hidden states", fontsize=10, y=1)
    plt.show()


if __name__ == "__main__":
    fire.Fire(main)